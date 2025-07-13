import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from mistral_inference.generate import generate
from dual_moe import MaxExpertsList, ExpertOutput


@dataclass
class AssistantConfig:
    """Assistant 설정 정보"""
    assistant_id: str
    prompt_template: str
    classification_threshold: float
    expert_affinity: Dict[int, float]  # expert_id: affinity_score 매핑


@dataclass
class AssistantOutput:
    """Assistant 출력 정보"""
    assistant_id: str
    comment_text: str
    confidence_score: float
    related_experts: List[int]


class AssistantRouter:
    """Assistant 시스템 라우터"""
    
    def __init__(self, tokenizer, model, max_assistants: int = 4):
        self.tokenizer = tokenizer
        self.model = model
        self.max_assistants = max_assistants
        
        # 프롬프트 템플릿들
        self.base_prompt_templates = {
            'analytical': "다음 질문을 분석적 관점에서 보완하여 답변하세요: {question}\n원본 답변: {first_output}\n보완 관점:",
            'creative': "다음 질문을 창의적 관점에서 보완하여 답변하세요: {question}\n원본 답변: {first_output}\n보완 관점:",
            'practical': "다음 질문을 실용적 관점에서 보완하여 답변하세요: {question}\n원본 답변: {first_output}\n보완 관점:",
            'critical': "다음 질문을 비판적 관점에서 보완하여 답변하세요: {question}\n원본 답변: {first_output}\n보완 관점:"
        }
        
        # 성능 모니터링
        self.processing_time = 0.0
        self.assistant_count = 0
    
    def _cluster_experts_by_activation(self, max_experts: MaxExpertsList) -> Dict[int, List[ExpertOutput]]:
        """Expert들을 활성화 패턴별로 클러스터링"""
        if len(max_experts.experts) <= 1:
            return {0: max_experts.experts}
        
        # 활성화 점수를 기반으로 클러스터링
        activation_scores = np.array([expert.activation_score for expert in max_experts.experts]).reshape(-1, 1)
        
        # 클러스터 수 결정 (최대 Assistant 수와 Expert 수 중 작은 값)
        n_clusters = min(self.max_assistants, len(max_experts.experts))
        
        if n_clusters == 1:
            return {0: max_experts.experts}
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(activation_scores)
        
        # 클러스터별로 Expert 그룹화
        expert_clusters = {}
        for i, expert in enumerate(max_experts.experts):
            cluster_id = cluster_labels[i]
            if cluster_id not in expert_clusters:
                expert_clusters[cluster_id] = []
            expert_clusters[cluster_id].append(expert)
        
        return expert_clusters
    
    def _generate_prompt_template(self, expert_group: List[ExpertOutput]) -> str:
        """Expert 그룹 특성에 따른 프롬프트 템플릿 생성"""
        # 평균 활성화 점수로 템플릿 선택
        avg_activation = sum(expert.activation_score for expert in expert_group) / len(expert_group)
        
        if avg_activation > 0.8:
            return self.base_prompt_templates['analytical']
        elif avg_activation > 0.6:
            return self.base_prompt_templates['creative']
        elif avg_activation > 0.4:
            return self.base_prompt_templates['practical']
        else:
            return self.base_prompt_templates['critical']
    
    def _calculate_threshold(self, expert_group: List[ExpertOutput]) -> float:
        """Expert 그룹 기반 분류 임계값 계산"""
        if not expert_group:
            return 0.5
        
        # 활성화 점수의 평균을 기반으로 임계값 설정
        avg_activation = sum(expert.activation_score for expert in expert_group) / len(expert_group)
        
        # 임계값은 평균 활성화 점수의 70%로 설정
        return avg_activation * 0.7
    
    def _create_assistants(self, max_experts: MaxExpertsList) -> List[AssistantConfig]:
        """max_experts 기반으로 Assistant 생성"""
        assistants = []
        
        # 1. Expert 활성화 패턴 분석
        expert_clusters = self._cluster_experts_by_activation(max_experts)
        
        # 2. 각 클러스터별로 Assistant 생성
        for cluster_id, expert_group in expert_clusters.items():
            assistant_config = AssistantConfig(
                assistant_id=f"assistant_{cluster_id}",
                prompt_template=self._generate_prompt_template(expert_group),
                classification_threshold=self._calculate_threshold(expert_group),
                expert_affinity={}
            )
            
            # 3. Expert와의 친화도 계산
            for expert_output in expert_group:
                assistant_config.expert_affinity[expert_output.expert_id] = expert_output.activation_score
            
            assistants.append(assistant_config)
        
        return assistants
    
    def _build_comment_prompt(self, prompt_template: str, original_question: str, 
                            first_output: torch.Tensor, expert_affinity: Dict[int, float]) -> str:
        """코멘트 생성을 위한 프롬프트 구성"""
        # 텐서를 텍스트로 변환 (간단한 예시)
        first_output_text = f"[Expert 출력 - 차원: {first_output.shape}]"
        
        # 친화도 정보 추가
        affinity_info = ", ".join([f"Expert{expert_id}: {score:.3f}" 
                                  for expert_id, score in expert_affinity.items()])
        
        prompt = prompt_template.format(
            question=original_question,
            first_output=first_output_text
        )
        
        prompt += f"\n관련 Expert 정보: {affinity_info}\n"
        
        return prompt
    
    def _generate_comment(self, prompt: str, assistant_id: str) -> str:
        """실제 Generate 함수 호출"""
        try:
            # 1. 토크나이저로 프롬프트 인코딩
            tokenized_prompt = self.tokenizer.encode(prompt)
            
            # 2. Generate 함수 호출
            output_tokens, _ = generate(
                [tokenized_prompt],
                self.model,
                max_tokens=256,
                temperature=0.7,
                eos_id=self.tokenizer.eos_id
            )
            
            # 3. 디코딩
            if output_tokens and len(output_tokens) > 0:
                comment_text = self.tokenizer.decode(output_tokens[0])
            else:
                comment_text = "생성 실패"
            
            # 4. 코멘트 형태로 포맷팅
            formatted_comment = self._format_as_comment(comment_text, assistant_id)
            
            return formatted_comment
            
        except Exception as e:
            return f"[{assistant_id}] 생성 오류: {str(e)}"
    
    def _format_as_comment(self, raw_text: str, assistant_id: str) -> str:
        """생성된 텍스트를 코멘트 형태로 포맷팅"""
        # 원본 텍스트 정리
        cleaned_text = raw_text.strip()
        
        # 너무 긴 경우 자르기
        if len(cleaned_text) > 300:
            cleaned_text = cleaned_text[:300] + "..."
        
        return f"[{assistant_id}] 원래 질문에 대한 보완적 관점: {cleaned_text}"
    
    def _calculate_confidence(self, comment_output: str, expert_affinity: Dict[int, float]) -> float:
        """코멘트 출력 기반 신뢰도 점수 계산"""
        # 간단한 신뢰도 계산 (실제로는 더 복잡한 메트릭 사용)
        
        # 1. 텍스트 길이 기반 점수 (20-200자 범위에서 최적)
        text_length = len(comment_output)
        length_score = min(1.0, max(0.1, (text_length - 20) / 180))
        
        # 2. Expert 친화도 평균
        affinity_score = sum(expert_affinity.values()) / len(expert_affinity) if expert_affinity else 0.5
        
        # 3. 오류 감지
        error_penalty = 0.5 if "오류" in comment_output or "실패" in comment_output else 0.0
        
        # 4. 최종 신뢰도 계산
        confidence = (length_score * 0.4 + affinity_score * 0.6) - error_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def bridge_through_assistants(self, original_question: str, first_output: torch.Tensor, 
                                max_experts: MaxExpertsList) -> List[AssistantOutput]:
        """Assistant 시스템 메인 처리 함수"""
        import time
        start_time = time.time()
        
        # 1. 동적 Assistant 생성
        assistants = self._create_assistants(max_experts)
        self.assistant_count = len(assistants)
        
        # 2. 각 Assistant별 코멘트 생성
        assistant_outputs = []
        
        for assistant in assistants:
            # 2-1. 프롬프트 구성
            prompt = self._build_comment_prompt(
                assistant.prompt_template,
                original_question,
                first_output,
                assistant.expert_affinity
            )
            
            # 2-2. Generate 함수 호출
            comment_output = self._generate_comment(prompt, assistant.assistant_id)
            
            # 2-3. 신뢰도 점수 계산
            confidence_score = self._calculate_confidence(comment_output, assistant.expert_affinity)
            
            # 2-4. 관련 Expert 식별
            related_experts = [
                expert_id for expert_id, affinity in assistant.expert_affinity.items()
                if affinity > assistant.classification_threshold
            ]
            
            assistant_outputs.append(AssistantOutput(
                assistant_id=assistant.assistant_id,
                comment_text=comment_output,
                confidence_score=confidence_score,
                related_experts=related_experts
            ))
        
        self.processing_time = time.time() - start_time
        return assistant_outputs
    
    def get_performance_stats(self) -> Dict[str, float]:
        """성능 통계 반환"""
        return {
            'processing_time': self.processing_time,
            'assistant_count': self.assistant_count,
            'avg_time_per_assistant': self.processing_time / max(1, self.assistant_count)
        }
    
    def reset_performance_stats(self):
        """성능 통계 초기화"""
        self.processing_time = 0.0
        self.assistant_count = 0