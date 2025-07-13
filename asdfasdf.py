"""
DualMoEPipeline - 전체 시스템을 통합하는 메인 파이프라인

이 파일은 mistral-inference를 기반으로 하여 다음 기능들을 통합합니다:
1. 첫 번째 MoE 패스 (일반적인 Expert 선택)
2. Assistant 시스템 (선택된 Expert 기반 코멘트 생성)
3. 임베딩 처리 (Assistant 출력 벡터화)
4. 두 번째 MoE 패스 (재라우팅)
5. Fusion 처리 (출력 결합)
6. 설문 시스템 (피드백 수집)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import time
import numpy as np
from pathlib import Path

# mistral-inference 기본 모듈들
from mistral_inference.model import Transformer
from mistral_inference.tokenizer import MistralTokenizer
from mistral_inference.generate import generate


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
    """Fusion 처리 결과 메트릭"""
    expert_id: int
    similarity_score: float
    novelty_score: float
    fusion_degree: float
    last_updated: datetime


@dataclass
class PipelineResult:
    """파이프라인 전체 결과"""
    user_query: str
    first_output: torch.Tensor
    second_output: torch.Tensor
    fused_output: torch.Tensor
    assistant_outputs: List[AssistantOutput]
    fusion_metrics: List[FusionMetrics]
    max_experts: List[ExpertOutput]
    processing_time: float
    survey_response: Optional[Dict] = None


class DualMoEPipeline:
    """
    전체 Dual MoE 시스템을 통합하는 메인 파이프라인 클래스
    
    이 클래스는 mistral-inference의 기본 구조를 확장하여
    이중 MoE 시스템을 구현합니다.
    """
    
    def __init__(
        self,
        model: Transformer,
        tokenizer: MistralTokenizer,
        max_experts_count: int = 8,
        assistant_threshold: float = 0.1,
        fusion_adjustment_rates: Dict[str, float] = None
    ):
        """
        파이프라인 초기화
        
        Args:
            model: 로드된 Mistral 모델
            tokenizer: Mistral 토크나이저
            max_experts_count: 최대 Expert 수
            assistant_threshold: Assistant 생성 임계값
            fusion_adjustment_rates: Fusion degree 조정 비율
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_experts_count = max_experts_count
        self.assistant_threshold = assistant_threshold
        
        # Fusion degree 저장소
        self.fusion_degrees: Dict[int, float] = {}
        
        # 조정 비율 설정
        self.fusion_adjustment_rates = fusion_adjustment_rates or {
            'positive': 0.05,  # 긍정적 피드백 시 증가량
            'negative': -0.08  # 부정적 피드백 시 감소량
        }
        
        # 설문 응답 저장소
        self.survey_responses: List[Dict] = []
        
        # 통계 정보
        self.processing_stats = {
            'total_queries': 0,
            'avg_processing_time': 0.0,
            'fusion_updates': 0
        }
        
        print(f"DualMoEPipeline initialized with {max_experts_count} max experts")
    
    def process_query(
        self, 
        user_query: str, 
        enable_survey: bool = False,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> PipelineResult:
        """
        사용자 쿼리를 처리하는 메인 함수
        
        Args:
            user_query: 사용자 입력 질문
            enable_survey: 설문 조사 활성화 여부
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            PipelineResult: 전체 처리 결과
        """
        start_time = time.time()
        
        try:
            # 1. 입력 전처리
            input_tensor = self._preprocess_query(user_query)
            
            # 2. 첫 번째 MoE 패스
            print("=== First MoE Pass ===")
            first_output, max_experts = self._forward_first_pass(input_tensor)
            
            # 3. Assistant 시스템 처리
            print("=== Assistant System Processing ===")
            assistant_outputs = self._process_assistants(
                user_query, first_output, max_experts, max_tokens, temperature
            )
            
            # 4. 임베딩 처리
            print("=== Embedding Processing ===")
            self._vectorize_assistant_outputs(assistant_outputs)
            
            # 5. 두 번째 MoE 패스
            print("=== Second MoE Pass ===")
            second_output = self._forward_second_pass(assistant_outputs, max_experts)
            
            # 6. Fusion 처리
            print("=== Fusion Processing ===")
            fusion_metrics = self._calculate_fusion_metrics(
                first_output, second_output, max_experts
            )
            fused_output = self._apply_fusion_weights(
                first_output, second_output, fusion_metrics
            )
            
            # 7. 설문 조사 (옵션)
            survey_response = None
            if enable_survey:
                print("=== Survey Collection ===")
                survey_response = self._collect_survey_response(
                    user_query, first_output, second_output, fused_output
                )
            
            # 8. 결과 패키징
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                user_query=user_query,
                first_output=first_output,
                second_output=second_output,
                fused_output=fused_output,
                assistant_outputs=assistant_outputs,
                fusion_metrics=fusion_metrics,
                max_experts=max_experts,
                processing_time=processing_time,
                survey_response=survey_response
            )
            
            # 9. 통계 업데이트
            self._update_processing_stats(processing_time)
            
            print(f"Pipeline completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"Error in pipeline processing: {str(e)}")
            raise
    
    def _preprocess_query(self, user_query: str) -> torch.Tensor:
        """
        사용자 쿼리를 모델 입력 형태로 전처리
        
        Args:
            user_query: 사용자 입력 문자열
            
        Returns:
            torch.Tensor: 모델 입력 텐서
        """
        # 토크나이징
        tokenized = self.tokenizer.encode(user_query)
        
        # 텐서 변환
        input_tensor = torch.tensor([tokenized], dtype=torch.long)
        
        # 모델의 임베딩 레이어 통과
        with torch.no_grad():
            embedded = self.model.embed_tokens(input_tensor)
        
        return embedded
    
    def _forward_first_pass(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[ExpertOutput]]:
        """
        첫 번째 MoE 패스 실행
        
        Args:
            input_tensor: 입력 텐서
            
        Returns:
            Tuple[torch.Tensor, List[ExpertOutput]]: 첫 번째 출력과 max_experts
        """
        # 기본 Mistral 모델 forward 실행
        with torch.no_grad():
            # 각 레이어별로 처리하여 Expert 정보 추출
            hidden_states = input_tensor
            all_expert_outputs = []
            
            for i, layer in enumerate(self.model.layers):
                # 레이어 실행
                layer_output = layer(hidden_states)
                
                # Expert 활성화 정보 시뮬레이션 (실제 구현에서는 MoE 레이어에서 추출)
                expert_outputs = self._simulate_expert_selection(layer_output, i)
                all_expert_outputs.extend(expert_outputs)
                
                hidden_states = layer_output
            
            # 최종 출력
            first_output = self.model.norm(hidden_states)
            
            # 상위 Expert들 선택
            max_experts = self._select_max_experts(all_expert_outputs)
        
        print(f"First pass completed. Selected {len(max_experts)} experts")
        return first_output, max_experts
    
    def _simulate_expert_selection(self, layer_output: torch.Tensor, layer_idx: int) -> List[ExpertOutput]:
        """
        Expert 선택 시뮬레이션 (실제 MoE 레이어가 없는 경우)
        
        Args:
            layer_output: 레이어 출력
            layer_idx: 레이어 인덱스
            
        Returns:
            List[ExpertOutput]: 시뮬레이션된 Expert 출력들
        """
        expert_outputs = []
        
        # 가상의 Expert 8개 생성
        for expert_id in range(8):
            # 활성화 점수 시뮬레이션
            activation_score = torch.rand(1).item()
            
            # 임계값 이상인 Expert만 선택
            if activation_score > self.assistant_threshold:
                expert_output = ExpertOutput(
                    expert_id=layer_idx * 8 + expert_id,  # 고유 ID 생성
                    output_tensor=layer_output,
                    weight=activation_score,
                    activation_score=activation_score,
                    fusion_degree=self.fusion_degrees.get(expert_id, 1.0)
                )
                expert_outputs.append(expert_output)
        
        return expert_outputs
    
    def _select_max_experts(self, all_expert_outputs: List[ExpertOutput]) -> List[ExpertOutput]:
        """
        상위 Expert들 선택
        
        Args:
            all_expert_outputs: 모든 Expert 출력들
            
        Returns:
            List[ExpertOutput]: 선택된 상위 Expert들
        """
        # 활성화 점수 기준 정렬
        sorted_experts = sorted(
            all_expert_outputs, 
            key=lambda x: x.activation_score, 
            reverse=True
        )
        
        # 상위 N개 선택
        max_experts = sorted_experts[:self.max_experts_count]
        
        return max_experts
    
    def _process_assistants(
        self, 
        user_query: str, 
        first_output: torch.Tensor, 
        max_experts: List[ExpertOutput],
        max_tokens: int,
        temperature: float
    ) -> List[AssistantOutput]:
        """
        Assistant 시스템 처리
        
        Args:
            user_query: 원본 질문
            first_output: 첫 번째 출력
            max_experts: 선택된 Expert들
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            List[AssistantOutput]: Assistant 출력들
        """
        assistant_outputs = []
        
        # Expert 클러스터링
        expert_clusters = self._cluster_experts(max_experts)
        
        # 각 클러스터별 Assistant 생성
        for cluster_id, expert_group in expert_clusters.items():
            try:
                # 프롬프트 생성
                prompt = self._build_assistant_prompt(user_query, expert_group)
                
                # Generate 함수 호출
                comment_text = self._generate_assistant_comment(
                    prompt, max_tokens, temperature
                )
                
                # 신뢰도 점수 계산
                confidence_score = self._calculate_confidence_score(
                    comment_text, expert_group
                )
                
                # 관련 Expert 식별
                related_experts = [exp.expert_id for exp in expert_group]
                
                assistant_output = AssistantOutput(
                    assistant_id=f"assistant_{cluster_id}",
                    comment_text=comment_text,
                    confidence_score=confidence_score,
                    related_experts=related_experts
                )
                
                assistant_outputs.append(assistant_output)
                
            except Exception as e:
                print(f"Error processing assistant {cluster_id}: {str(e)}")
                continue
        
        print(f"Generated {len(assistant_outputs)} assistant comments")
        return assistant_outputs
    
    def _cluster_experts(self, max_experts: List[ExpertOutput]) -> Dict[int, List[ExpertOutput]]:
        """
        Expert들을 클러스터링
        
        Args:
            max_experts: Expert 출력들
            
        Returns:
            Dict[int, List[ExpertOutput]]: 클러스터별 Expert 그룹
        """
        # 간단한 클러스터링: 활성화 점수 기준으로 그룹화
        clusters = {}
        
        # 활성화 점수 기준 3개 그룹으로 분할
        sorted_experts = sorted(max_experts, key=lambda x: x.activation_score, reverse=True)
        
        cluster_size = len(sorted_experts) // 3 + 1
        
        for i in range(0, len(sorted_experts), cluster_size):
            cluster_id = i // cluster_size
            clusters[cluster_id] = sorted_experts[i:i + cluster_size]
        
        return clusters
    
    def _build_assistant_prompt(self, user_query: str, expert_group: List[ExpertOutput]) -> str:
        """
        Assistant용 프롬프트 구성
        
        Args:
            user_query: 원본 질문
            expert_group: 관련 Expert 그룹
            
        Returns:
            str: 구성된 프롬프트
        """
        # Expert 정보 요약
        expert_info = []
        for expert in expert_group:
            expert_info.append(f"Expert {expert.expert_id} (활성화: {expert.activation_score:.3f})")
        
        expert_summary = ", ".join(expert_info)
        
        # 프롬프트 템플릿
        prompt = f"""다음 질문에 대해 보완적인 관점에서 코멘트를 작성해주세요.

원본 질문: {user_query}

관련 Expert 정보: {expert_summary}

이 Expert들의 특성을 고려하여, 원본 질문에 대한 새로운 관점이나 추가적인 고려사항을 제시해주세요.
답변은 간결하고 구체적으로 작성해주세요.

보완적 관점:"""
        
        return prompt
    
    def _generate_assistant_comment(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Assistant 코멘트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            
        Returns:
            str: 생성된 코멘트
        """
        try:
            # 프롬프트 토크나이징
            tokenized_prompt = self.tokenizer.encode(prompt)
            
            # Generate 함수 호출
            output_tokens, _ = generate(
                [tokenized_prompt],
                self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                eos_id=self.tokenizer.eos_id
            )
            
            # 디코딩
            generated_text = self.tokenizer.decode(output_tokens[0])
            
            # 프롬프트 부분 제거
            if prompt in generated_text:
                comment_text = generated_text.split(prompt)[-1].strip()
            else:
                comment_text = generated_text.strip()
            
            return comment_text
            
        except Exception as e:
            print(f"Error generating assistant comment: {str(e)}")
            return f"코멘트 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _calculate_confidence_score(self, comment_text: str, expert_group: List[ExpertOutput]) -> float:
        """
        신뢰도 점수 계산
        
        Args:
            comment_text: 생성된 코멘트
            expert_group: 관련 Expert 그룹
            
        Returns:
            float: 신뢰도 점수 (0-1)
        """
        # 간단한 신뢰도 계산: 텍스트 길이와 Expert 활성화 점수 기반
        text_length_score = min(len(comment_text) / 100, 1.0)  # 텍스트 길이 기준
        
        # Expert 활성화 점수 평균
        avg_activation = sum(exp.activation_score for exp in expert_group) / len(expert_group)
        
        # 종합 신뢰도
        confidence_score = (text_length_score * 0.3 + avg_activation * 0.7)
        
        return confidence_score
    
    def _vectorize_assistant_outputs(self, assistant_outputs: List[AssistantOutput]) -> None:
        """
        Assistant 출력들을 벡터화
        
        Args:
            assistant_outputs: Assistant 출력들
        """
        for assistant_output in assistant_outputs:
            try:
                # 텍스트 임베딩 생성
                embedding_vector = self._create_text_embedding(assistant_output.comment_text)
                assistant_output.embedding_vector = embedding_vector
                
            except Exception as e:
                print(f"Error vectorizing assistant {assistant_output.assistant_id}: {str(e)}")
                # 기본 벡터 할당
                assistant_output.embedding_vector = torch.zeros(self.model.embed_tokens.embedding_dim)
    
    def _create_text_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 입력 텍스트
            
        Returns:
            torch.Tensor: 임베딩 벡터
        """
        # 토크나이징
        tokens = self.tokenizer.encode(text)
        
        # 텐서 변환
        token_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # 임베딩 생성
        with torch.no_grad():
            token_embeddings = self.model.embed_tokens(token_tensor)
            
            # 평균 풀링으로 문장 임베딩 생성
            sentence_embedding = token_embeddings.mean(dim=1).squeeze(0)
        
        return sentence_embedding
    
    def _forward_second_pass(
        self, 
        assistant_outputs: List[AssistantOutput], 
        max_experts: List[ExpertOutput]
    ) -> torch.Tensor:
        """
        두 번째 MoE 패스 실행
        
        Args:
            assistant_outputs: Assistant 출력들
            max_experts: 선택된 Expert들
            
        Returns:
            torch.Tensor: 두 번째 출력
        """
        # Assistant 임베딩들을 결합
        combined_embedding = self._combine_assistant_embeddings(assistant_outputs)
        
        # 결합된 임베딩을 모델에 통과
        with torch.no_grad():
            # 입력 형태로 변환
            input_tensor = combined_embedding.unsqueeze(0).unsqueeze(0)
            
            # 모델 실행
            hidden_states = input_tensor
            
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)
            
            # 최종 출력
            second_output = self.model.norm(hidden_states)
        
        return second_output
    
    def _combine_assistant_embeddings(self, assistant_outputs: List[AssistantOutput]) -> torch.Tensor:
        """
        Assistant 임베딩들을 결합
        
        Args:
            assistant_outputs: Assistant 출력들
            
        Returns:
            torch.Tensor: 결합된 임베딩
        """
        if not assistant_outputs:
            # 기본 임베딩 반환
            return torch.zeros(self.model.embed_tokens.embedding_dim)
        
        # 신뢰도 기반 가중 평균
        weighted_sum = torch.zeros_like(assistant_outputs[0].embedding_vector)
        total_weight = 0.0
        
        for assistant_output in assistant_outputs:
            if assistant_output.embedding_vector is not None:
                weight = assistant_output.confidence_score
                weighted_sum += weight * assistant_output.embedding_vector
                total_weight += weight
        
        if total_weight > 0:
            combined_embedding = weighted_sum / total_weight
        else:
            combined_embedding = torch.zeros_like(assistant_outputs[0].embedding_vector)
        
        return combined_embedding
    
    def _calculate_fusion_metrics(
        self, 
        first_output: torch.Tensor, 
        second_output: torch.Tensor, 
        max_experts: List[ExpertOutput]
    ) -> List[FusionMetrics]:
        """
        Fusion 메트릭 계산
        
        Args:
            first_output: 첫 번째 출력
            second_output: 두 번째 출력
            max_experts: 선택된 Expert들
            
        Returns:
            List[FusionMetrics]: Fusion 메트릭들
        """
        fusion_metrics = []
        
        for expert_output in max_experts:
            # 유사도 점수 계산
            similarity_score = self._calculate_similarity(first_output, second_output)
            
            # 참신성 점수 계산
            novelty_score = self._calculate_novelty(first_output, second_output)
            
            # 현재 fusion_degree 가져오기
            current_fusion_degree = self.fusion_degrees.get(expert_output.expert_id, 1.0)
            
            # 새로운 fusion_degree 계산
            new_fusion_degree = self._adjust_fusion_degree(
                current_fusion_degree, 
                similarity_score, 
                novelty_score, 
                expert_output.activation_score
            )
            
            # 업데이트
            self.fusion_degrees[expert_output.expert_id] = new_fusion_degree
            expert_output.fusion_degree = new_fusion_degree
            
            fusion_metrics.append(FusionMetrics(
                expert_id=expert_output.expert_id,
                similarity_score=similarity_score,
                novelty_score=novelty_score,
                fusion_degree=new_fusion_degree,
                last_updated=datetime.now()
            ))
        
        return fusion_metrics
    
    def _calculate_similarity(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """
        두 출력 간 유사도 계산
        
        Args:
            first_output: 첫 번째 출력
            second_output: 두 번째 출력
            
        Returns:
            float: 유사도 점수 (0-1)
        """
        # 코사인 유사도 계산
        first_flat = first_output.flatten()
        second_flat = second_output.flatten()
        
        similarity = F.cosine_similarity(first_flat, second_flat, dim=0)
        
        # 0-1 범위로 정규화
        similarity_score = (similarity.item() + 1) / 2
        
        return similarity_score
    
    def _calculate_novelty(self, first_output: torch.Tensor, second_output: torch.Tensor) -> float:
        """
        참신성 점수 계산
        
        Args:
            first_output: 첫 번째 출력
            second_output: 두 번째 출력
            
        Returns:
            float: 참신성 점수 (0-1)
        """
        # 유사도의 역수로 참신성 계산
        similarity = self._calculate_similarity(first_output, second_output)
        novelty_score = 1.0 - similarity
        
        return novelty_score
    
    def _adjust_fusion_degree(
        self, 
        current_degree: float, 
        similarity: float, 
        novelty: float, 
        activation: float
    ) -> float:
        """
        Fusion degree 조정
        
        Args:
            current_degree: 현재 fusion degree
            similarity: 유사도 점수
            novelty: 참신성 점수
            activation: 활성화 점수
            
        Returns:
            float: 조정된 fusion degree
        """
        # 조정 공식
        novelty_factor = novelty * 0.6
        similarity_factor = (1.0 - similarity) * 0.3
        activation_factor = activation * 0.1
        
        adjustment = novelty_factor + similarity_factor + activation_factor
        
        # 현재 degree에 조정값 적용
        new_degree = current_degree + (adjustment - 0.5) * 0.1
        
        # 범위 제한 (0-1)
        new_degree = max(0.0, min(1.0, new_degree))
        
        return new_degree
    
    def _apply_fusion_weights(
        self, 
        first_output: torch.Tensor, 
        second_output: torch.Tensor, 
        fusion_metrics: List[FusionMetrics]
    ) -> torch.Tensor:
        """
        Fusion 가중치 적용
        
        Args:
            first_output: 첫 번째 출력
            second_output: 두 번째 출력
            fusion_metrics: Fusion 메트릭들
            
        Returns:
            torch.Tensor: 융합된 출력
        """
        # 전체 fusion 가중치 계산
        total_fusion_weight = sum(metric.fusion_degree for metric in fusion_metrics)
        
        if total_fusion_weight == 0:
            return first_output
        
        # 정규화된 가중치 계산
        alpha = total_fusion_weight / len(fusion_metrics)
        beta = 1.0 - alpha
        
        # 가중 결합
        fused_output = beta * first_output + alpha * second_output
        
        return fused_output
    
    def _collect_survey_response(
        self, 
        user_query: str, 
        first_output: torch.Tensor, 
        second_output: torch.Tensor, 
        fused_output: torch.Tensor
    ) -> Dict:
        """
        설문 응답 수집
        
        Args:
            user_query: 사용자 질문
            first_output: 첫 번째 출력
            second_output: 두 번째 출력
            fused_output: 융합된 출력
            
        Returns:
            Dict: 설문 응답 정보
        """
        survey_response = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'relevance_scores': [],
            'feedback_text': '',
            'outputs': {
                'first': self._tensor_to_text(first_output),
                'second': self._tensor_to_text(second_output),
                'fused': self._tensor_to_text(fused_output)
            }
        }
        
        # 실제 구현에서는 사용자 인터페이스를 통해 점수 수집
        # 여기서는 시뮬레이션
        print("\n=== 설문 조사 ===")
        print(f"질문: {user_query}")
        print(f"첫 번째 출력: {survey_response['outputs']['first'][:100]}...")
        print(f"두 번째 출력: {survey_response['outputs']['second'][:100]}...")
        print(f"융합된