"""
DualMoE Pipeline Implementation based on mistral-inference
Implements dual-pass MoE with Assistant system and fusion degree calculation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import logging
from pathlib import Path

# mistral-inference imports
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


@dataclass
class ExpertOutput:
    """Expert 출력 정보를 담는 데이터 클래스"""
    expert_id: int
    output_tensor: torch.Tensor
    weight: float
    activation_score: float
    fusion_degree: float = 1.0


@dataclass
class AssistantOutput:
    """Assistant 출력 정보를 담는 데이터 클래스"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]
    embedding_vector: Optional[torch.Tensor] = None


@dataclass
class FusionMetrics:
    """Fusion 계산 결과를 담는 데이터 클래스"""
    expert_id: int
    similarity_score: float
    novelty_score: float
    fusion_degree: float
    last_updated: datetime


@dataclass
class SurveyResponse:
    """설문 응답 데이터 클래스"""
    question: str
    relevance_scores: List[int]  # 1-5 점수
    timestamp: datetime
    user_id: Optional[str] = None


class MaxExpertsList:
    """최대 활성화된 Expert들의 리스트"""
    def __init__(self, max_count: int = 8):
        self.max_count = max_count
        self.experts: List[ExpertOutput] = []
        self.threshold = 0.1  # 최소 활성화 임계값
    
    def add_expert(self, expert_output: ExpertOutput):
        if expert_output.activation_score > self.threshold:
            self.experts.append(expert_output)
            # 활성화 점수 기준으로 정렬하고 상위 max_count개만 유지
            self.experts.sort(key=lambda x: x.activation_score, reverse=True)
            self.experts = self.experts[:self.max_count]
    
    def get_expert_ids(self) -> List[int]:
        return [expert.expert_id for expert in self.experts]


class DualOutputMoE(nn.Module):
    """Dual-pass MoE 시스템"""
    
    def __init__(self, model_layers, gate_network, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.model_layers = model_layers
        self.gate_network = gate_network
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert 네트워크들 (기존 모델 레이어 활용)
        self.experts = nn.ModuleList([
            layer for layer in model_layers[:num_experts]
        ])
        
        self.logger = logging.getLogger(__name__)
    
    def forward_first_pass(self, input_tensor: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, MaxExpertsList]:
        """
        첫 번째 패스: 일반적인 MoE 출력 + max_experts 생성
        
        Args:
            input_tensor: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            first_output: [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList
        """
        batch_size, seq_len, hidden_dim = input_tensor.shape
        
        # 1. 게이트 네트워크로 라우팅 점수 계산
        gate_input = input_tensor.reshape(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        routing_logits = self.gate_network(gate_input)  # [batch_size * seq_len, num_experts]
        routing_scores = F.softmax(routing_logits, dim=-1)
        
        # 2. Top-K 선택
        top_k_scores, top_k_indices = torch.topk(routing_scores, self.top_k, dim=-1)
        
        # 3. 각 Expert 실행
        expert_outputs = []
        max_experts = MaxExpertsList()
        
        for expert_id in range(self.num_experts):
            # Expert에 해당하는 토큰들 선택
            expert_mask = (top_k_indices == expert_id).any(dim=-1)
            if expert_mask.sum() == 0:
                continue
            
            expert_input = gate_input[expert_mask]
            if expert_input.shape[0] == 0:
                continue
                
            # Expert 실행
            expert_output = self.experts[expert_id](expert_input.unsqueeze(1)).squeeze(1)
            
            # 가중치 계산
            expert_weights = routing_scores[:, expert_id][expert_mask]
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # 활성화 점수 계산
            activation_score = routing_scores[:, expert_id].mean().item()
            
            expert_output_obj = ExpertOutput(
                expert_id=expert_id,
                output_tensor=weighted_output,
                weight=expert_weights.mean().item(),
                activation_score=activation_score
            )
            
            expert_outputs.append(expert_output_obj)
            max_experts.add_expert(expert_output_obj)
        
        # 4. 가중합으로 최종 출력 계산
        first_output = torch.zeros_like(input_tensor)
        for expert_output in expert_outputs:
            # 출력을 원래 형태로 복원
            expert_contribution = torch.zeros_like(gate_input)
            expert_mask = (top_k_indices == expert_output.expert_id).any(dim=-1)
            expert_contribution[expert_mask] = expert_output.output_tensor
            
            expert_contribution = expert_contribution.reshape(batch_size, seq_len, hidden_dim)
            first_output += expert_contribution
        
        self.logger.info(f"First pass completed with {len(max_experts.experts)} active experts")
        return first_output, max_experts
    
    def forward_second_pass(self, assistant_embedded_vector: torch.Tensor, max_experts: MaxExpertsList) -> torch.Tensor:
        """
        두 번째 패스: Assistant 출력 기반 재라우팅
        
        Args:
            assistant_embedded_vector: [batch_size, seq_len, hidden_dim]
            max_experts: MaxExpertsList
        
        Returns:
            second_output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = assistant_embedded_vector.shape
        
        # 1. Assistant 벡터 기반 새로운 라우팅 점수 계산
        gate_input = assistant_embedded_vector.reshape(-1, hidden_dim)
        routing_logits = self.gate_network(gate_input)
        assistant_routing_scores = F.softmax(routing_logits, dim=-1)
        
        # 2. max_experts 리스트 내에서만 재라우팅
        second_output = torch.zeros_like(assistant_embedded_vector)
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 새로운 가중치 계산 (fusion_degree 적용)
            new_weight = assistant_routing_scores[:, expert_id].mean().item()
            adjusted_weight = new_weight * expert_output.fusion_degree
            
            # Expert 재실행
            expert_input = gate_input
            expert_layer_output = self.experts[expert_id](expert_input.unsqueeze(1)).squeeze(1)
            
            # 가중치 적용
            weighted_output = expert_layer_output * adjusted_weight
            weighted_output = weighted_output.reshape(batch_size, seq_len, hidden_dim)
            
            second_output += weighted_output
        
        self.logger.info(f"Second pass completed with {len(max_experts.experts)} filtered experts")
        return second_output


class AssistantRouter:
    """Assistant 시스템 라우터"""
    
    def __init__(self, tokenizer: MistralTokenizer, model: Transformer, classification_threshold: float = 0.3):
        self.tokenizer = tokenizer
        self.model = model
        self.classification_threshold = classification_threshold
        self.logger = logging.getLogger(__name__)
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """Expert들을 활성화 패턴으로 클러스터링"""
        # 간단한 구현: 활성화 점수 기준으로 그룹화
        clusters = {}
        
        for i, expert in enumerate(max_experts.experts):
            cluster_id = i // 2  # 2개씩 그룹화
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(expert)
        
        return clusters
    
    def _generate_prompt_template(self, expert_group: List[ExpertOutput]) -> str:
        """Expert 그룹 기반 프롬프트 템플릿 생성"""
        expert_ids = [expert.expert_id for expert in expert_group]
        return f"""
        당신은 Expert {expert_ids}의 전문성을 가진 AI Assistant입니다.
        다음 질문에 대해 보완적인 관점에서 코멘트를 작성해주세요.
        
        질문: {{question}}
        
        기존 답변의 맥락을 고려하여 새로운 통찰이나 관점을 제공해주세요.
        코멘트는 간결하고 명확하게 작성해주세요.
        """
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """실제 Generate 함수 호출"""
        try:
            # 1. 프롬프트 토크나이징
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=prompt)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # 2. Generate 함수 호출
            out_tokens, _ = generate(
                [tokens],
                self.model,
                max_tokens=256,
                temperature=0.7,
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
            
            # 3. 디코딩
            comment_text = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
            
            # 4. 코멘트 포맷팅
            formatted_comment = f"[Assistant_{assistant_id}] 보완적 관점: {comment_text}"
            
            return formatted_comment
            
        except Exception as e:
            self.logger.error(f"Error generating comment: {e}")
            return f"[Assistant_{assistant_id}] 코멘트 생성 중 오류가 발생했습니다."
    
    def bridge_through_assistants(self, original_question: str, first_output: torch.Tensor, max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """Assistant 시스템 메인 처리 함수"""
        self.logger.info(f"Processing question through {len(max_experts.experts)} experts")
        
        # 1. Expert 클러스터링
        expert_clusters = self._cluster_experts_by_activation(max_experts)
        
        # 2. 각 클러스터별 Assistant 생성 및 코멘트 생성
        assistant_outputs = []
        
        for cluster_id, expert_group in expert_clusters.items():
            # Assistant 설정
            assistant_id = f"cluster_{cluster_id}"
            prompt_template = self._generate_prompt_template(expert_group)
            
            # 프롬프트 구성
            prompt = prompt_template.format(question=original_question)
            
            # 코멘트 생성
            comment_text = self._generate_comment(prompt, assistant_id)
            
            # 신뢰도 점수 계산 (활성화 점수 평균)
            confidence_score = sum(expert.activation_score for expert in expert_group) / len(expert_group)
            
            # 관련 Expert 리스트
            related_experts = [expert.expert_id for expert in expert_group 
                             if expert.activation_score > self.classification_threshold]
            
            assistant_output = AssistantOutput(
                assistant_id=assistant_id,
                comment_text=comment_text,
                confidence_score=confidence_score,
                related_experts=related_experts
            )
            
            assistant_outputs.append(assistant_output)
        
        self.logger.info(f"Generated {len(assistant_outputs)} assistant comments")
        return assistant_outputs


class EmbeddingProcessor:
    """임베딩 처리 시스템"""
    
    def __init__(self, model: Transformer, tokenizer: MistralTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def _create_embedding(self, text: str) -> torch.Tensor:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            # 토크나이징
            completion_request = ChatCompletionRequest(
                messages=[UserMessage(content=text)]
            )
            tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
            
            # 임베딩 레이어 통과
            with torch.no_grad():
                token_embeddings = self.model.tok_embeddings(torch.tensor(tokens))
                # 평균 풀링으로 문장 임베딩 생성
                sentence_embedding = token_embeddings.mean(dim=0)
            
            return sentence_embedding
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            # 기본값 반환
            return torch.zeros(self.model.args.dim)
    
    def vectorize_assistant_output(self, assistant_outputs: List[AssistantOutput]) -> List[AssistantOutput]:
        """Assistant 출력들을 벡터화"""
        for assistant_output in assistant_outputs:
            embedding_vector = self._create_embedding(assistant_output.comment_text)
            assistant_output.embedding_vector = embedding_vector
        
        self.logger.info(f"Vectorized {len(assistant_outputs)} assistant outputs")
        return assistant_outputs
    
    def route_to_experts(self, assistant_outputs: List[AssistantOutput], max_experts: MaxExpertsList) -> torch.Tensor:
        """임베딩된 Assistant 출력을 Expert 시스템으로 라우팅"""
        # 모든 Assistant 임베딩을 가중평균으로 결합
        if not assistant_outputs:
            return torch.zeros(1, 1, self.model.args.dim)
        
        combined_embedding = torch.zeros_like(assistant_outputs[0].embedding_vector)
        total_weight = 0
        
        for assistant_output in assistant_outputs:
            if assistant_output.embedding_vector is not None:
                weight = assistant_output.confidence_score
                combined_embedding += assistant_output.embedding_vector * weight
                total_weight += weight
        
        if total_weight > 0:
            combined_embedding /= total_weight
        
        # 배치 차원 추가
        routing_vector = combined_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        return routing_vector


class FusionController:
    """Fusion 제어 시스템"""
    
    def __init__(self):
        self.fusion_degrees: Dict[int, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def _calculate_output_similarity(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """두 출력 간의 유사도 계산"""
        # 코사인 유사도 계산
        first_flat = first_output.reshape(-1)
        second_flat = second_output.reshape(-1)
        
        similarity = F.cosine_similarity(first_flat, second_flat, dim=0)
        return similarity.item()
    
    def _calculate_novelty_score(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """참신성 점수 계산"""
        # L2 거리 기반 참신성 계산
        l2_distance = torch.norm(first_output - second_output, p=2)
        max_distance = torch.norm(first_output, p=2) + torch.norm(second_output, p=2)
        
        if max_distance == 0:
            return 0.0
        
        novelty_score = (l2_distance / max_distance).item()
        return min(1.0, novelty_score)
    
    def calculate_fusion_degree(self, first_output: torch.Tensor, second_output: torch.Tensor, max_experts: MaxExpertsList) -> List[FusionMetrics]:
        """두 출력 간 비교를 통한 fusion_degree 계산"""
        fusion_metrics = []
        
        # 전체 출력 간 유사도 및 참신성 계산
        similarity_score = self._calculate_output_similarity(first_output, second_output)
        novelty_score = self._calculate_novelty_score(first_output, second_output)
        
        for expert_output in max_experts.experts:
            expert_id = expert_output.expert_id
            
            # 기존 fusion_degree 가져오기
            current_fusion_degree = self.fusion_degrees.get(expert_id, 1.0)
            
            # 동적 조정 공식
            new_fusion_degree = self._adjust_fusion_degree(
                current_fusion_degree,
                similarity_score,
                novelty_score,
                expert_output.activation_score
            )
            
            fusion_metric = FusionMetrics(
                expert_id=expert_id,
                similarity_score=similarity_score,
                novelty_score=novelty_score,
                fusion_degree=new_fusion_degree,
                last_updated=datetime.now()
            )
            
            fusion_metrics.append(fusion_metric)
            
            # 업데이트된 fusion_degree 저장
            self.fusion_degrees[expert_id] = new_fusion_degree
            expert_output.fusion_degree = new_fusion_degree
        
        self.logger.info(f"Calculated fusion degrees for {len(fusion_metrics)} experts")
        return fusion_metrics
    
    def _adjust_fusion_degree(self, current_degree: float, similarity: float, novelty: float, activation: float) -> float:
        """동적 fusion_degree 조정 공식"""
        # 높은 참신성 + 적당한 유사도 = 높은 fusion_degree
        # 낮은 참신성 + 높은 유사도 = 낮은 fusion_degree
        
        novelty_factor = novelty * 0.6  # 참신성 가중치
        similarity_factor = (1.0 - similarity) * 0.3  # 차이점 가중치
        activation_factor = activation * 0.1  # 활성화 가중치
        
        adjustment = novelty_factor + similarity_factor + activation_factor
        
        # 현재 degree에 조정값 적용
        new_degree = current_degree + (adjustment - 0.5) * 0.1
        
        # 범위 제한 (0-1)
        new_degree = max(0.0, min(1.0, new_degree))
        
        return new_degree
    
    def apply_fusion_weights(self, first_output: torch.Tensor, second_output: torch.Tensor, fusion_metrics: List[FusionMetrics]) -> torch.Tensor:
        """Fusion degree를 적용한 최종 출력 생성"""
        if not fusion_metrics:
            return first_output
        
        # 전체 fusion 가중치 계산
        total_fusion_weight = sum(metric.fusion_degree for metric in fusion_metrics)
        
        if total_fusion_weight == 0:
            return first_output
        
        # 정규화된 가중치 계산
        alpha = total_fusion_weight / len(fusion_metrics)  # 평균 fusion degree
        beta = 1.0 - alpha  # 원본 출력 영향도
        
        # 최종 출력 생성
        fused_output = beta * first_output + alpha * second_output
        
        self.logger.info(f"Applied fusion with alpha={alpha:.3f}, beta={beta:.3f}")
        return fused_output


class SurveySystem:
    """설문 조사 시스템"""
    
    def __init__(self, fusion_controller: FusionController):
        self.fusion_controller = fusion_controller
        self.survey_responses: List[SurveyResponse] = []
        self.adjustment_rates = {
            'positive': 0.05,  # 긍정적 피드백 시 증가율
            'negative': -0.1   # 부정적 피드백 시 감소율 (더 강하게)
        }
        self.logger = logging.getLogger(__name__)
    
    def collect_survey_response(self, question: str, relevance_scores: List[int], user_id: Optional[str] = None) -> SurveyResponse:
        """사용자로부터 설문 응답 수집"""
        survey_response = SurveyResponse(
            question=question,
            relevance_scores=relevance_scores,
            timestamp=datetime.now(),
            user_id=user_id
        )
        
        self.survey_responses.append(survey_response)
        self.logger.info(f"Collected survey response with scores: {relevance_scores}")
        
        return survey_response
    
    def update_fusion_degrees(self, recent_responses: List[SurveyResponse]) -> Dict[int, float]:
        """설문 결과를 기반으로 fusion_degree 업데이트"""
        updated_degrees = {}
        
        for response in recent_responses:
            avg_relevance = sum(response.relevance_scores) / len(response.relevance_scores)
            
            # 평균 점수 기반 조정
            if avg_relevance >= 4.0:  # 긍정적 피드백
                adjustment_rate = self.adjustment_rates['positive']
            elif avg_relevance <= 2.0:  # 부정적 피드백
                adjustment_rate = self.adjustment_rates['negative']
            else:  # 중립
                adjustment_rate = 0.0
            
            # 모든 Expert의 fusion_degree 업데이트
            for expert_id in self.fusion_controller.fusion_degrees:
                current_degree = self.fusion_controller.fusion_degrees[expert_id]
                new_degree = max(0.0, min(1.0, current_degree + adjustment_rate))
                self.fusion_controller.fusion_degrees[expert_id] = new_degree
                updated_degrees[expert_id] = new_degree
        
        self.logger.info(f"Updated fusion degrees based on {len(recent_responses)} responses")
        return updated_degrees


class DualMoEPipeline:
    """DualMoE 파이프라인 메인 클래스"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        # 모델 로딩
        self.tokenizer = MistralTokenizer.from_file(tokenizer_path)
        self.model = Transformer.from_folder(model_path)
        
        # 시스템 초기화
        self.experts_system = DualOutputMoE(
            model_layers=self.model.layers,
            gate_network=self.model.layers[0].feed_forward,  # 임시로 FFN 사용
            num_experts=8,
            top_k=2
        )
        
        self.assistants_system = AssistantRouter(self.tokenizer, self.model)
        self.embedding_processor = EmbeddingProcessor(self.model, self.tokenizer)
        self.fusion_controller = FusionController()
        self.survey_system = SurveySystem(self.fusion_controller)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def process_query(self, user_query: str, enable_survey: bool = False) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        self.logger.info(f"Processing query: {user_query}")
        
        # 1. 입력 전처리
        input_tensor = self._preprocess_query(user_query)
        
        # 2. 첫 번째 MoE 패스
        first_output, max_experts = self.experts_system.forward_first_pass(input_tensor)
        
        # 3. Assistant 시스템 처리
        assistant_outputs = self.assistants_system.bridge_through_assistants(
            user_query, first_output, max_experts
        )
        
        # 4. 임베딩 처리
        embedded_outputs = self.embedding_processor.vectorize_assistant_output(assistant_outputs)
        
        # 5. 두 번째 MoE 패스
        routing_vector = self.embedding_processor.route_to_experts(embedded_outputs, max_experts)
        second_output = self.experts_system.forward_second_pass(routing_vector, max_experts)
        
        # 6. Fusion 처리
        fusion_metrics = self.fusion_controller.calculate_fusion_degree(
            first_output, second_output, max_experts
        )
        
        fused_output = self.fusion_controller.apply_fusion_weights(
            first_output, second_output, fusion_metrics
        )
        
        # 7. 결과 패키징
        result = {
            'user_query': user_query,
            'first_output': first_output,
            'second_output': second_output,
            'fused_output': fused_output,
            'assistant_outputs': assistant_outputs,
            'fusion_metrics': fusion_metrics,
            'max_experts': max_experts,
            'survey_ready': enable_survey
        }
        
        self.logger.info("Pipeline processing completed")
        return result
    
    def _preprocess_query(self, query: str) -> torch.Tensor:
        """쿼리 전처리"""
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=query)]
        )
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens
        
        # 임베딩으로 변환
        with torch.no_grad():
            input_tensor = self.model.tok_embeddings(torch.tensor(tokens)).unsqueeze(0)
        
        return input_tensor
    
    def submit_survey(self, relevance_scores: List[int], user_id: Optional[str] = None) -> Dict[int, float]:
        """설문 제출 및 fusion_degree 업데이트"""
        if not hasattr(self, '_last_query'):
            raise ValueError("No recent query to evaluate")
        
        survey_response = self.survey_system.collect_survey_response(
            self._last_query, relevance_scores, user_id
        )
        
        updated_degrees = self.survey_system.update_fusion_degrees([survey_response])
        
        return updated_degrees


# 사용 예시
if __name__ == "__main__":
    # 모델 경로 설정
    model_path = "./mistral-nemo-instruct-v0.1"  # 실제 모델 경로로 변경
    tokenizer_path = "./mistral-nemo-instruct-v0.1/tekken.json"  # 실제 토크나이저 경로로 변경
    
    try:
        # 파이프라인 초기화
        pipeline = DualMoEPipeline(model_path, tokenizer_path)
        
        # 쿼리 처리
        user_query = "인공지능의 미래 발전 방향에 대해 설명해주세요."
        result = pipeline.process_query(user_query, enable_survey=True)
        
        # 결과 출력
        print(f"Query: {result['user_query']}")
        print(f"Active Experts: {len(result['max_experts'].experts)}")
        print(f"Assistant Comments: {len(result['assistant_outputs'])}")
        
        for assistant_output in result['assistant_outputs']:
            print(f"- {assistant_output.comment_text}")
        
        # 설문 제출 예시
        relevance_scores = [4, 5, 3, 4, 5]  # 1-5 점수
        updated_degrees = pipeline.submit_survey(relevance_scores)
        print(f"Updated fusion degrees: {updated_degrees}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure mistral-inference is installed and model paths are correct")