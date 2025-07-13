"""
Dual Output MoE System for Mistral-based Architecture
이중 패스 MoE 시스템 - 첫 번째 패스에서 max_experts 생성, 두 번째 패스에서 Assistant 피드백 기반 재라우팅

기존 Mistral MoE 구조를 확장하여 두 번의 출력을 생성하는 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np


@dataclass
class ExpertOutput:
    """Expert 출력 정보를 저장하는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0
    
    def to_dict(self):
        return {
            'expert_id': self.expert_id,
            'weight': self.weight,
            'activation_score': self.activation_score,
            'fusion_degree': self.fusion_degree
        }


@dataclass
class MaxExpertsList:
    """max_experts 리스트를 관리하는 클래스"""
    experts: List[ExpertOutput]
    max_count: int = 8
    threshold: float = 0.1
    
    def __post_init__(self):
        if not self.experts:
            self.experts = []
    
    def add_expert(self, expert_output: ExpertOutput):
        """Expert 추가 (정렬된 상태 유지)"""
        self.experts.append(expert_output)
        self.experts.sort(key=lambda x: x.activation_score, reverse=True)
        
        # 최대 개수 제한
        if len(self.experts) > self.max_count:
            self.experts = self.experts[:self.max_count]
    
    def get_top_experts(self, k: int = None) -> List[ExpertOutput]:
        """상위 k개 Expert 반환"""
        if k is None:
            k = len(self.experts)
        return self.experts[:k]
    
    def filter_by_threshold(self) -> List[ExpertOutput]:
        """임계값 이상의 Expert만 반환"""
        return [e for e in self.experts if e.activation_score >= self.threshold]


class DualOutputMoE(nn.Module):
    """
    이중 출력 MoE 시스템
    - 첫 번째 패스: 일반적인 MoE 출력 + max_experts 생성
    - 두 번째 패스: Assistant 피드백 기반 재라우팅 출력
    """
    
    def __init__(self, experts: List[nn.Module], gate: nn.Module, 
                 args, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = len(experts)
        self.top_k = top_k
        self.args = args
        
        # 내부 상태 관리
        self.max_experts_cache = None
        self.fusion_degrees = {}  # expert_id -> fusion_degree 매핑
        
        # 메트릭 수집
        self.metrics = {
            'first_pass_calls': 0,
            'second_pass_calls': 0,
            'expert_utilization': torch.zeros(self.num_experts),
            'routing_entropy': []
        }
    
    def forward_first_pass(self, input_tensor: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, MaxExpertsList]:
        """
        첫 번째 패스: 일반적인 MoE 출력 + max_experts 생성
        
        Args:
            input_tensor: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] (선택적)
        
        Returns:
            first_output: [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList
        """
        self.metrics['first_pass_calls'] += 1
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        # 1. 입력 텐서 reshape (MoE 처리를 위해)
        input_reshaped = input_tensor.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # 2. 게이트 네트워크로 라우팅 점수 계산
        routing_logits = self.gate(input_reshaped)  # [batch_size * seq_len, num_experts]
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # 3. Top-K 선택
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # 4. 라우팅 엔트로피 계산 (다양성 메트릭)
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1)
        self.metrics['routing_entropy'].append(entropy.mean().item())
        
        # 5. 각 Expert 실행 및 출력 수집
        expert_outputs = []
        final_output = torch.zeros_like(input_reshaped)
        
        for expert_idx in range(self.num_experts):
            # 해당 Expert가 선택된 토큰들 찾기
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.sum() > 0:
                # Expert 실행
                expert_input = input_reshaped[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                
                # 가중치 계산
                expert_token_indices = torch.where(expert_mask)[0]
                expert_weights = []
                
                for token_idx in expert_token_indices:
                    # 해당 토큰에서 현재 Expert의 가중치 찾기
                    expert_positions = (top_k_indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_positions) > 0:
                        expert_weights.append(top_k_probs[token_idx, expert_positions[0]].item())
                    else:
                        expert_weights.append(0.0)
                
                # 평균 가중치 및 활성화 점수 계산
                avg_weight = np.mean(expert_weights) if expert_weights else 0.0
                activation_score = routing_probs[:, expert_idx].mean().item()
                
                # Expert 출력 저장
                expert_outputs.append(ExpertOutput(
                    expert_id=expert_idx,
                    output_tensor=expert_output,
                    weight=avg_weight,
                    activation_score=activation_score,
                    fusion_degree=self.fusion_degrees.get(expert_idx, 1.0)
                ))
                
                # 최종 출력에 기여도 추가
                for i, token_idx in enumerate(expert_token_indices):
                    final_output[token_idx] += expert_output[i] * expert_weights[i]
                
                # 활용도 메트릭 업데이트
                self.metrics['expert_utilization'][expert_idx] += expert_mask.sum().item()
        
        # 6. max_experts 리스트 생성
        max_experts = MaxExpertsList(experts=[], max_count=8, threshold=0.1)
        
        # 활성화 점수 기준으로 정렬하여 상위 Expert들 선택
        sorted_experts = sorted(expert_outputs, key=lambda x: x.activation_score, reverse=True)
        for expert_output in sorted_experts:
            if expert_output.activation_score >= max_experts.threshold:
                max_experts.add_expert(expert_output)
        
        # 7. 결과 reshape
        first_output = final_output.view(batch_size, seq_len, hidden_dim)
        
        # 8. 캐시 저장
        self.max_experts_cache = max_experts
        
        return first_output, max_experts
    
    def forward_second_pass(self, assistant_embedded_vector: torch.Tensor, 
                           max_experts: MaxExpertsList) -> torch.Tensor:
        """
        두 번째 패스: Assistant 출력 기반 재라우팅
        
        Args:
            assistant_embedded_vector: [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList
        
        Returns:
            second_output: [batch_size, seq_len, hidden_dim]
        """
        self.metrics['second_pass_calls'] += 1
        batch_size, seq_len, hidden_dim = assistant_embedded_vector.shape
        
        # 1. 입력 텐서 reshape
        input_reshaped = assistant_embedded_vector.view(-1, hidden_dim)
        
        # 2. Assistant 벡터 기반 새로운 라우팅 점수 계산
        assistant_routing_logits = self.gate(input_reshaped)
        assistant_routing_probs = F.softmax(assistant_routing_logits, dim=-1)
        
        # 3. max_experts 리스트 내에서만 재라우팅
        filtered_experts = []
        total_weight = 0.0
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 새로운 라우팅 가중치 계산
            new_routing_weight = assistant_routing_probs[:, expert_id].mean().item()
            
            # Fusion degree 적용
            adjusted_weight = new_routing_weight * expert_output.fusion_degree
            total_weight += adjusted_weight
            
            # 업데이트된 Expert 출력 생성
            updated_expert = ExpertOutput(
                expert_id=expert_id,
                output_tensor=expert_output.output_tensor,
                weight=adjusted_weight,
                activation_score=expert_output.activation_score,
                fusion_degree=expert_output.fusion_degree
            )
            
            filtered_experts.append(updated_expert)
        
        # 4. 가중치 정규화
        if total_weight > 0:
            for expert in filtered_experts:
                expert.weight = expert.weight / total_weight
        
        # 5. 재가중합으로 최종 출력 계산
        second_output = torch.zeros_like(input_reshaped)
        
        for expert in filtered_experts:
            if expert.weight > 0:
                # Expert 출력을 올바른 차원으로 브로드캐스팅
                expert_contribution = self._broadcast_expert_output(
                    expert.output_tensor, input_reshaped.shape, expert.weight
                )
                second_output += expert_contribution
        
        # 6. 결과 reshape
        second_output = second_output.view(batch_size, seq_len, hidden_dim)
        
        return second_output
    
    def _broadcast_expert_output(self, expert_output: torch.Tensor, 
                               target_shape: torch.Size, weight: float) -> torch.Tensor:
        """
        Expert 출력을 타겟 형태로 브로드캐스팅
        
        Args:
            expert_output: Expert의 출력 텐서
            target_shape: 타겟 형태 [total_tokens, hidden_dim]
            weight: 가중치
        
        Returns:
            브로드캐스트된 출력 텐서
        """
        total_tokens, hidden_dim = target_shape
        
        if expert_output.shape[0] == total_tokens:
            # 이미 올바른 형태
            return expert_output * weight
        else:
            # 평균 풀링 후 브로드캐스팅
            avg_output = expert_output.mean(dim=0, keepdim=True)  # [1, hidden_dim]
            return avg_output.expand(total_tokens, hidden_dim) * weight
    
    def update_fusion_degree(self, expert_id: int, new_degree: float):
        """
        특정 Expert의 fusion_degree 업데이트
        
        Args:
            expert_id: Expert ID
            new_degree: 새로운 fusion degree (0.0 ~ 1.0)
        """
        self.fusion_degrees[expert_id] = max(0.0, min(1.0, new_degree))
        
        # 캐시된 max_experts도 업데이트
        if self.max_experts_cache:
            for expert in self.max_experts_cache.experts:
                if expert.expert_id == expert_id:
                    expert.fusion_degree = new_degree
                    break
    
    def get_expert_metrics(self) -> Dict:
        """
        Expert 활용도 및 성능 메트릭 반환
        
        Returns:
            메트릭 딕셔너리
        """
        return {
            'first_pass_calls': self.metrics['first_pass_calls'],
            'second_pass_calls': self.metrics['second_pass_calls'],
            'expert_utilization': self.metrics['expert_utilization'].tolist(),
            'routing_entropy_avg': np.mean(self.metrics['routing_entropy']) if self.metrics['routing_entropy'] else 0.0,
            'fusion_degrees': dict(self.fusion_degrees),
            'active_experts': len([d for d in self.fusion_degrees.values() if d > 0.1])
        }
    
    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = {
            'first_pass_calls': 0,
            'second_pass_calls': 0,
            'expert_utilization': torch.zeros(self.num_experts),
            'routing_entropy': []
        }
    
    def forward(self, input_tensor: torch.Tensor, 
                assistant_vector: Optional[torch.Tensor] = None,
                return_max_experts: bool = False) -> torch.Tensor:
        """
        통합 forward 함수 (기존 MoE 호환성 유지)
        
        Args:
            input_tensor: 입력 텐서
            assistant_vector: Assistant 벡터 (선택적)
            return_max_experts: max_experts 반환 여부
        
        Returns:
            출력 텐서 (또는 튜플)
        """
        if assistant_vector is None:
            # 첫 번째 패스만 실행
            first_output, max_experts = self.forward_first_pass(input_tensor)
            if return_max_experts:
                return first_output, max_experts
            return first_output
        else:
            # 두 번째 패스 실행
            if self.max_experts_cache is None:
                raise ValueError("두 번째 패스 실행 전에 첫 번째 패스가 실행되어야 합니다.")
            return self.forward_second_pass(assistant_vector, self.max_experts_cache)
    
    def save_state(self, filepath: str):
        """상태 저장"""
        state = {
            'fusion_degrees': self.fusion_degrees,
            'metrics': self.get_expert_metrics(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """상태 로드"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.fusion_degrees = state.get('fusion_degrees', {})
        # 문자열 키를 정수로 변환
        self.fusion_degrees = {int(k): v for k, v in self.fusion_degrees.items()}


# 사용 예시 및 테스트 코드
def create_test_dual_moe():
    """테스트용 DualOutputMoE 생성"""
    
    # 더미 Expert 생성
    class DummyExpert(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            
        def forward(self, x):
            return self.linear(x)
    
    # 더미 Gate 생성
    class DummyGate(nn.Module):
        def __init__(self, hidden_dim, num_experts):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, num_experts)
            
        def forward(self, x):
            return self.linear(x)
    
    # 파라미터 설정
    hidden_dim = 512
    num_experts = 8
    
    # 컴포넌트 생성
    experts = [DummyExpert(hidden_dim) for _ in range(num_experts)]
    gate = DummyGate(hidden_dim, num_experts)
    
    # DualOutputMoE 생성
    args = type('Args', (), {'hidden_dim': hidden_dim})()
    dual_moe = DualOutputMoE(experts, gate, args, top_k=2)
    
    return dual_moe


if __name__ == "__main__":
    # 테스트 실행
    print("DualOutputMoE 테스트 시작...")
    
    # 1. 시스템 생성
    dual_moe = create_test_dual_moe()
    
    # 2. 테스트 입력
    batch_size, seq_len, hidden_dim = 2, 10, 512
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 3. 첫 번째 패스 실행
    print("첫 번째 패스 실행...")
    first_output, max_experts = dual_moe.forward_first_pass(input_tensor)
    print(f"첫 번째 출력 형태: {first_output.shape}")
    print(f"max_experts 개수: {len(max_experts.experts)}")
    
    # 4. 더미 Assistant 벡터 생성
    assistant_vector = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 5. 두 번째 패스 실행
    print("두 번째 패스 실행...")
    second_output = dual_moe.forward_second_pass(assistant_vector, max_experts)
    print(f"두 번째 출력 형태: {second_output.shape}")
    
    # 6. 메트릭 확인
    metrics = dual_moe.get_expert_metrics()
    print(f"메트릭: {metrics}")
    
    # 7. Fusion degree 업데이트 테스트
    dual_moe.update_fusion_degree(0, 0.5)
    print("Fusion degree 업데이트 완료")
    
    print("테스트 완료!")