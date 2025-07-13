import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class ExpertOutput:
    """Expert 출력 정보를 담는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


class MaxExpertsList:
    """상위 활성화 Expert들을 관리하는 클래스"""
    def __init__(self, max_count: int = 8):
        self.max_count = max_count
        self.experts: List[ExpertOutput] = []
        self.threshold = 0.1  # 최소 활성화 임계값
    
    def add_expert(self, expert_output: ExpertOutput):
        """Expert 추가 (활성화 점수 기준 정렬 유지)"""
        if expert_output.activation_score > self.threshold:
            self.experts.append(expert_output)
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            if len(self.experts) > self.max_count:
                self.experts = self.experts[:self.max_count]
    
    def get_expert_ids(self) -> List[int]:
        """Expert ID 리스트 반환"""
        return [expert.expert_id for expert in self.experts]
    
    def get_expert_by_id(self, expert_id: int) -> Optional[ExpertOutput]:
        """ID로 Expert 검색"""
        for expert in self.experts:
            if expert.expert_id == expert_id:
                return expert
        return None


class DualOutputMoE(nn.Module):
    """두 번의 패스를 지원하는 MoE 시스템"""
    
    def __init__(self, experts: List[nn.Module], gate: nn.Module, top_k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.top_k = top_k
        self.num_experts = len(experts)
        
        # 성능 모니터링
        self.first_pass_time = 0.0
        self.second_pass_time = 0.0
    
    def _select_top_k(self, routing_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-K Expert 선택"""
        # routing_scores: [batch_size, seq_len, num_experts]
        top_k_values, top_k_indices = torch.topk(routing_scores, self.top_k, dim=-1)
        
        # 소프트맥스 적용
        top_k_weights = torch.softmax(top_k_values, dim=-1)
        
        return top_k_indices, top_k_weights
    
    def _weighted_sum(self, expert_outputs: List[ExpertOutput]) -> torch.Tensor:
        """Expert 출력들의 가중합 계산"""
        if not expert_outputs:
            return torch.zeros_like(expert_outputs[0].output_tensor)
        
        total_weight = sum(expert.weight for expert in expert_outputs)
        if total_weight == 0:
            return torch.zeros_like(expert_outputs[0].output_tensor)
        
        weighted_output = torch.zeros_like(expert_outputs[0].output_tensor)
        for expert in expert_outputs:
            normalized_weight = expert.weight / total_weight
            weighted_output += normalized_weight * expert.output_tensor
        
        return weighted_output
    
    def forward_first_pass(self, input_tensor: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, MaxExpertsList]:
        """
        첫 번째 패스: 일반적인 MoE 출력 + max_experts 생성
        
        Args:
            input_tensor: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] (선택사항)
        
        Returns:
            first_output: [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList
        """
        start_time = time.time()
        
        # 1. 게이트 네트워크로 라우팅 점수 계산
        routing_scores = self.gate(input_tensor)  # [batch_size, seq_len, num_experts]
        
        # 2. Top-K 선택
        top_k_indices, top_k_weights = self._select_top_k(routing_scores)
        
        # 3. 각 Expert 실행
        expert_outputs = []
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                for k in range(self.top_k):
                    expert_id = top_k_indices[batch_idx, seq_idx, k].item()
                    weight = top_k_weights[batch_idx, seq_idx, k].item()
                    
                    # Expert 실행
                    expert_input = input_tensor[batch_idx:batch_idx+1, seq_idx:seq_idx+1, :]
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # 활성화 점수 계산 (해당 위치에서의 라우팅 점수)
                    activation_score = routing_scores[batch_idx, seq_idx, expert_id].item()
                    
                    expert_outputs.append(ExpertOutput(
                        expert_id=expert_id,
                        output_tensor=expert_output,
                        weight=weight,
                        activation_score=activation_score
                    ))
        
        # 4. 가중합으로 최종 출력 계산
        first_output = self._weighted_sum(expert_outputs)
        
        # 5. max_experts 리스트 생성
        max_experts = MaxExpertsList()
        
        # Expert별로 평균 활성화 점수 계산
        expert_scores = {}
        expert_counts = {}
        
        for expert_output in expert_outputs:
            expert_id = expert_output.expert_id
            if expert_id not in expert_scores:
                expert_scores[expert_id] = 0.0
                expert_counts[expert_id] = 0
            
            expert_scores[expert_id] += expert_output.activation_score
            expert_counts[expert_id] += 1
        
        # 평균 점수 계산 및 정렬
        averaged_experts = []
        for expert_id, total_score in expert_scores.items():
            avg_score = total_score / expert_counts[expert_id]
            
            # 대표 출력 선택 (첫 번째 출력 사용)
            representative_output = None
            for expert_output in expert_outputs:
                if expert_output.expert_id == expert_id:
                    representative_output = expert_output
                    break
            
            if representative_output:
                averaged_experts.append(ExpertOutput(
                    expert_id=expert_id,
                    output_tensor=representative_output.output_tensor,
                    weight=representative_output.weight,
                    activation_score=avg_score
                ))
        
        # 상위 experts 선택
        averaged_experts.sort(key=lambda x: x.activation_score, reverse=True)
        for expert in averaged_experts[:max_experts.max_count]:
            max_experts.add_expert(expert)
        
        self.first_pass_time = time.time() - start_time
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
        start_time = time.time()
        
        # 1. Assistant 벡터 기반 새로운 라우팅 점수 계산
        assistant_routing_scores = self.gate(assistant_embedded_vector)
        
        # 2. max_experts 리스트 내에서만 재라우팅
        filtered_experts = []
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 새로운 가중치 계산
            new_weight = assistant_routing_scores[:, :, expert_id].mean().item()
            
            # Fusion degree 적용
            adjusted_weight = new_weight * expert_output.fusion_degree
            
            # Expert 재실행
            expert_new_output = self.experts[expert_id](assistant_embedded_vector)
            
            filtered_experts.append(ExpertOutput(
                expert_id=expert_id,
                output_tensor=expert_new_output,
                weight=adjusted_weight,
                activation_score=expert_output.activation_score,
                fusion_degree=expert_output.fusion_degree
            ))
        
        # 3. 재가중합으로 최종 출력
        second_output = self._weighted_sum(filtered_experts)
        
        self.second_pass_time = time.time() - start_time
        return second_output
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        return {
            'first_pass_time': self.first_pass_time,
            'second_pass_time': self.second_pass_time,
            'total_time': self.first_pass_time + self.second_pass_time,
            'num_experts': self.num_experts,
            'top_k': self.top_k
        }
    
    def reset_performance_stats(self):
        """성능 통계 초기화"""
        self.first_pass_time = 0.0
        self.second_pass_time = 0.0