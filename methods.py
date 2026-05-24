"""
核心方法模块 (methods.py)
定义 BaseSolver 抽象基类和各种求解器实现

用于ACL论文: Bridging the Intent Gap: Multi-Faceted Intent Recognition and Inference-Time Prompt Optimization

包含方法:
1. VanillaSolver - Zero-shot 直接回答
2. CoTSolver - Chain-of-Thought 推理
3. SelfRefineSolver - 自我改进
4. IAROSolver (Our Method) - Intent-Aware Response Optimization
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import config
from llm_client import AsyncLLMClient, create_solver_client


# ========== 响应数据结构 ==========
@dataclass
class SolverResponse:
    """求解器响应"""
    method: str
    query: str
    response: str
    
    # IARO 特有字段
    intent_analysis: Optional[Dict[str, Any]] = None
    
    # 元数据
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""
    timestamp: str = ""
    
    # 中间结果 (用于调试)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ========== 抽象基类 ==========
class BaseSolver(ABC):
    """
    求解器抽象基类
    
    所有方法必须实现 solve() 方法
    """
    
    def __init__(
        self,
        client: Optional[AsyncLLMClient] = None,
        name: str = "BaseSolver",
    ):
        """
        初始化求解器
        
        Args:
            client: LLM客户端
            name: 方法名称
        """
        self.client = client or create_solver_client()
        self.name = name
    
    @abstractmethod
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """
        求解用户查询
        
        Args:
            query: 用户查询
            **kwargs: 额外参数
        
        Returns:
            SolverResponse
        """
        pass
    
    async def solve_batch(
        self,
        queries: List[str],
        show_progress: bool = True,
        **kwargs
    ) -> List[SolverResponse]:
        """
        批量求解
        
        Args:
            queries: 查询列表
            show_progress: 是否显示进度
            **kwargs: 额外参数
        
        Returns:
            响应列表
        """
        from tqdm.asyncio import tqdm_asyncio
        
        tasks = [self.solve(q, **kwargs) for q in queries]
        
        if show_progress:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Solving ({self.name})"
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"⚠ Query {i} failed: {r}")
                processed.append(SolverResponse(
                    method=self.name,
                    query=queries[i],
                    response=f"[ERROR] {str(r)}",
                    timestamp=datetime.now().isoformat(),
                ))
            else:
                processed.append(r)
        
        return processed


# ========== Vanilla Solver (Zero-shot) ==========
class VanillaSolver(BaseSolver):
    """
    Vanilla Zero-shot Solver
    
    直接将用户查询发送给模型，不做任何处理。
    这是最基础的 baseline，用于展示即使是 SOTA 模型，
    在 zero-shot 下也可能忽略隐式需求。
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's request directly and comprehensively."""
    
    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="Vanilla")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Zero-shot 直接回答"""
        import time
        start = time.time()
        
        response = await self.client.generate(
            prompt=query,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2000,
        )
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
        )


# ========== Chain-of-Thought Solver ==========
class CoTSolver(BaseSolver):
    """
    Chain-of-Thought Solver
    
    使用 CoT prompting 让模型逐步推理用户需求。
    预期弱点：会推理出显式步骤，但往往忽略"用户没说但必须做"的隐式约束。
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that thinks step by step.

When answering a request:
1. First, carefully analyze what the user is asking for
2. Think through each step of how to address their needs
3. Consider any relevant context or requirements
4. Provide a comprehensive response

Think step by step before giving your final answer."""
    
    COT_USER_PROMPT = """Please help me with the following request. Think step by step about what I need.

Request: {query}

First, analyze my request step by step, then provide your response."""
    
    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="CoT")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Chain-of-Thought 推理"""
        import time
        start = time.time()
        
        prompt = self.COT_USER_PROMPT.format(query=query)
        
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2500,
        )
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
        )


# ========== Few-Shot CoT Solver ==========
class FewShotCoTSolver(BaseSolver):
    """
    Few-Shot Chain-of-Thought Solver
    
    使用示例驱动的 CoT prompting，比 Zero-Shot CoT 更强。
    包含 3 个精心设计的示例，展示如何识别隐式需求。
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that thinks carefully about user requests.
When responding, first identify what the user explicitly asked for AND what they implicitly need."""

    FEW_SHOT_EXAMPLES = """Here are examples of thorough request analysis:

---
Example 1:
User: "Write a function to read a config file"
Thinking: The user explicitly wants to read a config file. Implicitly, they likely need:
- Error handling for missing files
- Support for common formats (JSON, YAML, etc.)
- Validation of config values
Response: [A function with error handling and format detection]

---
Example 2:  
User: "Help me send an email to my team about the meeting"
Thinking: Explicit: send an email about a meeting. Implicit needs:
- Professional tone appropriate for team communication
- Include essential meeting details (time, location, agenda)
- Clear call-to-action (RSVP, prepare materials, etc.)
Response: [A professional email template with placeholders]

---
Example 3:
User: "Create a script to delete old log files"
Thinking: Explicit: delete old log files. Implicit safety needs:
- Define "old" (age threshold should be configurable)
- Dry-run mode to preview before deleting
- Exclude important/active logs
- Confirmation before destructive action
Response: [A safe script with dry-run and confirmation]

---
Now analyze this request the same way:
"""

    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="FewShot-CoT")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Few-Shot CoT 推理"""
        import time
        start = time.time()
        
        prompt = f"""{self.FEW_SHOT_EXAMPLES}
User: "{query}"
Thinking:"""
        
        response = await self.client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2500,
        )
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
        )


# ========== Rephrase-and-Respond (RaR) Solver ==========
class RaRSolver(BaseSolver):
    """
    Rephrase-and-Respond Solver
    
    基于 Deng et al. (ICLR 2024) "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves"
    
    核心思想：让模型先重述问题，明确任务要求，再回答
    这是 IARO 的直接竞争对手：两者都是 2-step pipeline，但 RaR 是非结构化的重述，
    而 IARO 是结构化的意图分析。
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant."""
    
    REPHRASE_PROMPT = """Before answering, first rephrase the following request in your own words to ensure you understand it completely. Consider:
- What is the user explicitly asking for?
- What might they implicitly need?
- What format/style would be most helpful?

Original Request: {query}

Rephrase the request (be thorough but concise):"""
    
    RESPOND_PROMPT = """Based on your understanding of the request, provide a comprehensive response.

Original Request: {query}

Your Understanding: {rephrased}

Now provide your response:"""
    
    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="RaR")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Rephrase-and-Respond 两阶段生成"""
        import time
        start = time.time()
        
        steps = []
        
        # Step 1: Rephrase
        rephrase_prompt = self.REPHRASE_PROMPT.format(query=query)
        rephrased = await self.client.generate(
            prompt=rephrase_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=500,
        )
        steps.append({"step": "rephrase", "output": rephrased})
        
        # Step 2: Respond based on rephrased understanding
        respond_prompt = self.RESPOND_PROMPT.format(query=query, rephrased=rephrased)
        response = await self.client.generate(
            prompt=respond_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=2000,
        )
        steps.append({"step": "respond", "output": response[:500] + "..."})
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
            intermediate_steps=steps,
        )


# ========== CRITIC Solver ==========
class CRITICSolver(BaseSolver):
    """
    CRITIC Solver
    
    基于 Gou et al. (ICLR 2024) "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing"
    
    简化版：不使用工具，仅使用 self-critique 机制
    流程：生成 -> 批判 -> 修正
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant."""
    
    CRITIQUE_PROMPT = """Critically evaluate the following response. Be specific about:
1. Factual accuracy issues
2. Missing important information
3. Logical flaws or inconsistencies
4. Ways to improve completeness

Original Request: {query}

Response to Critique:
{response}

Provide specific, actionable criticism:"""
    
    REVISE_PROMPT = """Revise your response based on the critique.

Original Request: {query}

Original Response:
{response}

Critique:
{critique}

Provide an improved response that addresses all the issues:"""
    
    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="CRITIC")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """CRITIC 三阶段：生成 -> 批判 -> 修正"""
        import time
        start = time.time()
        
        steps = []
        
        # Step 1: Initial generation
        response = await self.client.generate(
            prompt=query,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1500,
        )
        steps.append({"step": "initial_generation", "output": response[:500]})
        
        # Step 2: Self-critique
        critique_prompt = self.CRITIQUE_PROMPT.format(query=query, response=response)
        critique = await self.client.generate(
            prompt=critique_prompt,
            system_prompt="You are a critical reviewer. Be specific and actionable.",
            temperature=0.3,
            max_tokens=500,
        )
        steps.append({"step": "critique", "output": critique})
        
        # Step 3: Revise
        revise_prompt = self.REVISE_PROMPT.format(query=query, response=response, critique=critique)
        revised_response = await self.client.generate(
            prompt=revise_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=2000,
        )
        steps.append({"step": "revision", "output": revised_response[:500]})
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=revised_response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
            intermediate_steps=steps,
        )


# ========== Self-Refine Solver ==========
class SelfRefineSolver(BaseSolver):
    """
    Self-Refine Solver
    
    让模型生成初始响应后自我改进。
    这是 IARO 的 naive baseline，展示通用的 "Please improve" 不如
    基于结构化意图分析的改进精准。
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant."""
    
    REFINE_PROMPT = """Review your previous response and improve it.

Original Request: {query}

Your Previous Response:
{previous_response}

Please provide an improved response that:
1. Is more comprehensive
2. Addresses any missed aspects
3. Is better structured
4. Handles edge cases

Improved Response:"""
    
    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="SelfRefine")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Self-Refine 两阶段生成"""
        import time
        start = time.time()
        
        steps = []
        
        # Step 1: Initial generation
        initial_response = await self.client.generate(
            prompt=query,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1500,
        )
        
        steps.append({
            "step": "initial_generation",
            "output": initial_response[:500] + "..." if len(initial_response) > 500 else initial_response,
        })
        
        # Step 2: Self-refine
        refine_prompt = self.REFINE_PROMPT.format(
            query=query,
            previous_response=initial_response,
        )
        
        refined_response = await self.client.generate(
            prompt=refine_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=2000,
        )
        
        steps.append({
            "step": "self_refine",
            "output": refined_response[:500] + "..." if len(refined_response) > 500 else refined_response,
        })
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=refined_response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
            intermediate_steps=steps,
        )


# ========== IARO Solver (Our Method) ==========
class IAROSolver(BaseSolver):
    """
    IARO Solver - Intent-Aware Response Optimization (Our Method)
    
    核心创新：基于结构化意图本体 (Intent Ontology) 的两阶段方法 + 验证驱动的自我修正
    
    架构:
    - Module 1 (Intent Analyst): 分析查询，输出结构化意图 JSON
    - Module 2 (Generator): 基于分析结果生成响应
    - Module 3 (Verifier & Refiner): 验证响应是否满足约束，如果不满足则触发修正循环
    """
    
    # ===== Intent Analyst Prompt (V2 - Context-Aware + Comprehensive Implicit Needs) =====
    ANALYST_SYSTEM_PROMPT = """You are an expert Intent Analyst. Your task is to understand BOTH explicit goals AND implicit needs from the user's request and context.

Your analysis should cover:
1. **Explicit Goals (G)**: What the user directly asks for
2. **Implicit Needs (N_imp)**: Requirements that are NOT stated but CRITICAL for a production-ready solution
3. **Style Constraints (S)**: Tone, format, and presentation expectations

## CRITICAL: Read the Context Carefully
The context contains crucial information about:
- The environment (financial, medical, security-critical, etc.)
- Regulations and compliance requirements (GDPR, HIPAA, PCI-DSS, etc.)
- Edge cases and error scenarios
- Professional standards and best practices

## Implicit Needs Guidelines:
- Identify **4-5 implicit needs** based on the context and domain
- Consider: safety, security, compliance, error handling, edge cases, scalability
- **Most implicit needs should be "blocker" type** - they are critical for real-world deployment
- Only mark as "enhancement" if truly optional

## Domain-Specific Considerations:
- **coding**: Error handling, input validation, security, logging, compliance
- **writing**: Tone appropriateness, audience considerations, legal/compliance concerns
- **planning**: Risk mitigation, contingency planning, stakeholder considerations
- **qa**: Accuracy, completeness, source verification

Output ONLY valid JSON. Be thorough in identifying implicit needs from context."""

    ANALYST_USER_PROMPT = """Analyze the following user request and extract the intent structure.

## User Request:
{query}

## Context (IMPORTANT - use this to infer implicit needs):
{context}

## Output Format (JSON):
{{
    "explicit_goals": [
        {{"id": "G1", "description": "..."}},
        ...
    ],
    "implicit_needs": [
        {{
            "id": "N1",
            "description": "...",
            "category": "functional|safety|quality|contextual",
            "importance": "critical|important|nice_to_have",
            "integration_type": "blocker|enhancement",
            "rationale": "Why this is needed based on context"
        }},
        ...
    ],
    "style_constraints": [
        {{"id": "S1", "description": "...", "type": "tone|format|length|audience"}},
        ...
    ],
    "domain": "coding|writing|planning|qa|other",
    "expected_format": "code|text|structured_plan|explanation",
    "complexity": "simple|moderate|complex"
}}

## CRITICAL Guidelines:
1. **Read the CONTEXT carefully** - it contains crucial environment information
2. **Identify 4-5 implicit needs** from context (regulatory, safety, edge cases, compliance)
3. **Most needs should be "blocker" type** - they are essential for production-ready solutions
4. Consider: security, compliance (GDPR/HIPAA/etc), error handling, edge cases, scalability
5. Only mark as "enhancement" if truly optional for basic functionality"""

    # ===== Generator Prompt (V2 - Comprehensive Implicit Needs Implementation) =====
    GENERATOR_SYSTEM_PROMPT = """You are an expert assistant. Your objective is to address BOTH explicit goals AND implicit needs comprehensively.

## Intent Analysis:
{intent_analysis}

## Response Priority Order

### Priority 1: EXPLICIT GOALS (100% completion required)
- Address EVERY explicit goal the user stated
- This is the PRIMARY success criterion

### Priority 2: IMPLICIT NEEDS (Critical for production-ready solutions)
- **Address ALL identified implicit needs in your response**
- These represent real-world requirements users implicitly expect
- Integrate blockers directly into the solution
- Integrate enhancements as practical recommendations

### Priority 3: Domain-Appropriate Format
- **coding** → Working code with proper error handling and validation
- **writing** → Professional text addressing all stakeholder concerns
- **planning** → Structured plan with risk mitigation
- **qa** → Comprehensive explanation

## Response Guidelines:
1. **Complete Coverage**: Address both explicit goals AND implicit needs
2. **Practical Implementation**: Don't just mention needs - actually address them
3. **Production-Ready**: Response should be deployable in real-world scenarios
4. **Context-Aware**: Consider the specific environment from the context

## IMPORTANT: Implicit needs are NOT optional extras. They represent critical requirements that distinguish a basic response from a production-ready solution.
"""

    GENERATOR_USER_PROMPT = """Respond to the user's request:

{query}

## Checklist (in order of priority):
1. ✓ Address ALL explicit goals the user stated
2. ✓ Use the appropriate format for the domain (code/text/plan)
3. ✓ Keep response focused and appropriately sized
4. ✓ Add critical implicit needs only if they enhance (not replace) the main response

Deliver a practical, complete response that directly answers what the user asked."""

    # ===== Verifier Prompt WITH Intent Expansion =====
    VERIFIER_PROMPT = """You are a Quality Assurance Verifier with Intent Expansion capability.

## Original Query:
{query}

## Identified Needs (from initial analysis):
{needs}

## Generated Response:
{response}

## Verification Task (IN ORDER OF PRIORITY):

### Step 1: EXPLICIT GOALS CHECK (Critical)
- Did the response address ALL explicit goals (G1, G2, etc.)?
- This is the PRIMARY success criterion

### Step 2: IMPLICIT NEEDS CHECK
- Were the identified implicit needs addressed?
- Rate coverage for each need

### Step 3: INTENT EXPANSION (Important!)
- **Actively scan for LATENT REQUIREMENTS** that were MISSED in the initial analysis
- Ask yourself: "What critical needs does this query context reveal that we haven't considered?"
- Focus on: safety issues, edge cases, compliance requirements, user experience gaps
- Only identify needs that are TRULY CRITICAL (would cause real problems if missed)

### Step 4: Format Appropriateness
- Does the response format match expectations?

Output JSON:
{{
    "explicit_goals_completion": {{
        "all_met": true/false,
        "details": [{{"goal_id": "G1", "addressed": true/false, "explanation": "..."}}]
    }},
    "format_appropriate": true/false,
    "verification_results": [
        {{"need_id": "N1", "addressed": true/false, "explanation": "..."}}
    ],
    "overall_coverage": 0.0-1.0,
    "critical_issues": ["..."],
    "missing_aspects": ["..."],
    "newly_identified_needs": [
        {{
            "description": "A critical implicit need that was missed in initial analysis",
            "category": "safety|functional|compliance|user_experience",
            "rationale": "Why this need is critical and was not initially identified",
            "integration_type": "blocker|enhancement"
        }}
    ],
    "suggestions": ["..."]
}}

IMPORTANT for Intent Expansion:
- Only add 1-3 truly critical needs that were OVERLOOKED
- Each new need must have clear rationale
- Focus on safety, compliance, and critical edge cases
- Do NOT add trivial or nice-to-have features"""

    # ===== Refiner Prompt (P1 Fix - Focused Refinement) =====
    REFINER_PROMPT = """You are an Expert Editor. Improve the response based on QA feedback.

## Original Request:
{query}

## Previous Response:
{previous_response}

## QA Feedback:
{feedback}

## Refinement Priority (in order):
1. **Fix any EXPLICIT GOAL gaps** - If any explicit goals were missed, add them
2. **Fix format issues** - Ensure response format matches expectations
3. **Fix critical issues** - Address any bugs or errors
4. **Keep length reasonable** - Do NOT significantly expand the response

## Guidelines:
- Do NOT add extensive new sections
- Do NOT over-engineer the solution
- Focus on completing what was asked, not adding more features
- Keep the response CONCISE

Output the improved response (similar length to original, not longer)."""

    def __init__(
        self,
        client: Optional[AsyncLLMClient] = None,
        enable_verification: bool = False,  # Default: 2-step base mode (no verification)
        max_refines: int = 0,                # Default: no refinement loop
        enable_intent_expansion: bool = False,  # Default: no expansion
        name: str = "IARO",
    ):
        """
        初始化 IARO Solver
        
        IARO 有两种模式:
        1. IARO-Base (default): 2-step pipeline (Analyze -> Generate), 高效且已达 SOTA
        2. IARO-Recursive: 启用 Verification + Refinement Loop，用于极难 case
        
        Args:
            client: LLM客户端
            enable_verification: 是否启用验证模块 (True 启用 Recursive 模式)
            max_refines: 最大修正轮数 (>0 表示启用修正循环)
            enable_intent_expansion: 是否启用意图扩展
            name: 方法名称
        """
        super().__init__(client, name=name)
        self.enable_verification = enable_verification
        self.max_refines = max_refines
        self.enable_intent_expansion = enable_intent_expansion
    
    async def _analyze_intent(self, query: str, context: str = "") -> Dict[str, Any]:
        """Step 1: Intent Analysis with Context"""
        prompt = self.ANALYST_USER_PROMPT.format(
            query=query,
            context=context if context else "No additional context provided."
        )
        
        try:
            intent_analysis = await self.client.generate_json(
                prompt=prompt,
                system_prompt=self.ANALYST_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=2000,
            )
            return intent_analysis
        except (ValueError, Exception) as e:
            # Fallback: return minimal intent analysis
            print(f"    ⚠ Intent analysis JSON parse failed, using fallback: {e}")
            return {
                "explicit_goals": [{"id": "G1", "description": query}],
                "implicit_needs": [],
                "context_analysis": {"domain": "general"},
            }
    
    async def _generate_response(
        self,
        query: str,
        intent_analysis: Dict[str, Any],
    ) -> str:
        """Step 2: Intent-Aware Generation"""
        analysis_text = self._format_intent_analysis(intent_analysis)
        
        system_prompt = self.GENERATOR_SYSTEM_PROMPT.format(
            intent_analysis=analysis_text
        )
        
        user_prompt = self.GENERATOR_USER_PROMPT.format(query=query)
        
        response = await self.client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1800,  # P1 Fix: Reduced from 2500 to control response length
        )
        return response
    
    # Verifier Prompt WITHOUT Intent Expansion (P1 Fix - Simplified)
    VERIFIER_PROMPT_NO_EXPANSION = """You are a Quality Assurance Verifier. Focus on EXPLICIT GOAL completion.

## Original Query:
{query}

## Identified Needs:
{needs}

## Generated Response:
{response}

## Verification Task:
1. Did the response address ALL EXPLICIT GOALS?
2. Is the response format appropriate for the request?

Output JSON:
{{
    "explicit_goals_completion": {{
        "all_met": true/false,
        "details": [{{"goal_id": "G1", "addressed": true/false}}]
    }},
    "format_appropriate": true/false,
    "overall_coverage": 0.0-1.0,
    "critical_issues": ["..."]
}}"""

    async def _verify_response(
        self,
        query: str,
        intent_analysis: Dict[str, Any],
        response: str,
    ) -> Dict[str, Any]:
        """Step 3: Verify Response Coverage"""
        all_needs = []
        for i, goal in enumerate(intent_analysis.get("explicit_goals", [])):
            if isinstance(goal, dict):
                all_needs.append(f"[Explicit] {goal.get('id', f'G{i+1}')}: {goal.get('description', str(goal))}")
            else:
                all_needs.append(f"[Explicit] G{i+1}: {goal}")
        for i, need in enumerate(intent_analysis.get("implicit_needs", [])):
            if isinstance(need, dict):
                all_needs.append(f"[Implicit] {need.get('id', f'N{i+1}')}: {need.get('description', str(need))}")
            else:
                all_needs.append(f"[Implicit] N{i+1}: {need}")
        
        # Choose prompt based on intent_expansion setting
        verifier_prompt = self.VERIFIER_PROMPT if self.enable_intent_expansion else self.VERIFIER_PROMPT_NO_EXPANSION
        
        prompt = verifier_prompt.format(
            query=query,
            needs="\n".join(all_needs),
            response=response[:3000],  # Increase context window
        )
        
        try:
            verification = await self.client.generate_json(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2000,  # Increased to avoid truncation
            )
            return verification
        except (ValueError, Exception) as e:
            # Fallback: assume verification passed to avoid crash
            print(f"    ⚠ Verification JSON parse failed, using fallback: {e}")
            return {
                "verification_results": [],
                "missing_aspects": [],
                "critical_issues": [],
                "overall_score": 8,
                "is_perfect": True,  # Skip refinement on parse failure
                "suggestions": [],
            }

    async def _refine_response(
        self,
        query: str,
        response: str,
        verification: Dict[str, Any],
    ) -> str:
        """Step 4: Refine Response based on Verification"""
        missing = verification.get("missing_aspects", [])
        critical = verification.get("critical_issues", [])
        suggestions = verification.get("suggestions", [])
        new_needs = verification.get("newly_identified_needs", [])
        
        # Helper to convert items to strings (handle both str and dict)
        def to_str(item):
            if isinstance(item, str):
                return item
            elif isinstance(item, dict):
                return item.get("description", item.get("issue", str(item)))
            return str(item)
        
        feedback_lines = []
        if critical:
            feedback_lines.append(f"CRITICAL ISSUES: {'; '.join(to_str(c) for c in critical)}")
        
        if new_needs:
            feedback_lines.append("\nNEWLY IDENTIFIED CRITICAL NEEDS (MUST ADDRESS):")
            for n in new_needs:
                if isinstance(n, dict):
                    desc = n.get("description", str(n))
                    rationale = n.get("rationale", "")
                    feedback_lines.append(f"- {desc} (Why: {rationale})")
                else:
                    feedback_lines.append(f"- {n}")
        
        if missing:
            feedback_lines.append(f"MISSING ASPECTS: {'; '.join(to_str(m) for m in missing)}")
            
        if suggestions:
            feedback_lines.append(f"SUGGESTIONS: {'; '.join(to_str(s) for s in suggestions)}")
            
        feedback = "\n".join(feedback_lines)
        
        prompt = self.REFINER_PROMPT.format(
            query=query,
            previous_response=response,
            feedback=feedback
        )
        
        refined_response = await self.client.generate(
            prompt=prompt,
            system_prompt="You are an expert editor aiming for perfection.",
            temperature=0.5,
            max_tokens=3000,
        )
        return refined_response
    
    def _format_intent_analysis(self, analysis: Dict[str, Any]) -> str:
        """格式化意图分析为可读文本"""
        lines = []
        
        # Explicit Goals
        lines.append("## Explicit Goals (MUST address FIRST):")
        for goal in analysis.get("explicit_goals", []):
            desc = goal.get('description', str(goal)) if isinstance(goal, dict) else goal
            lines.append(f"- {desc}")
        
        # Implicit Needs (Blocker vs Enhancement)
        blocker_needs = []
        enhancement_needs = []
        
        for need in analysis.get("implicit_needs", []):
            if isinstance(need, dict):
                itype = need.get("integration_type", "enhancement").lower()
                if itype == "blocker":
                    blocker_needs.append(need)
                else:
                    enhancement_needs.append(need)
            else:
                enhancement_needs.append({"description": str(need)})
        
        if blocker_needs:
            lines.append("\n## BLOCKER Needs (Integrate INTO core solution):")
            for need in blocker_needs:
                desc = need.get("description", str(need))
                lines.append(f"- [BLOCKER] {desc}")
        
        if enhancement_needs:
            lines.append("\n## ENHANCEMENT Needs (Add AFTER core solution):")
            for need in enhancement_needs:
                desc = need.get("description", str(need))
                lines.append(f"- [ENHANCE] {desc}")
        
        # Style
        lines.append("\n## Style Constraints:")
        for style in analysis.get("style_constraints", []):
            desc = style.get('description', str(style)) if isinstance(style, dict) else style
            lines.append(f"- {desc}")
            
        return "\n".join(lines)
    
    async def solve(self, query: str, context: str = "", **kwargs) -> SolverResponse:
        """IARO Iterative Solving Process with Context"""
        import time
        start = time.time()
        
        steps = []
        
        # 1. Intent Analysis WITH CONTEXT
        intent_analysis = await self._analyze_intent(query, context)
        steps.append({"step": "intent_analysis", "output": intent_analysis})
        
        # 2. Initial Generation
        response = await self._generate_response(query, intent_analysis)
        steps.append({"step": "generation", "output": response[:500] + "..."})
        
        # 3. Verification & Refinement Loop
        final_response = response
        
        if self.enable_verification:
            for i in range(self.max_refines + 1):
                # Verify
                verification = await self._verify_response(query, intent_analysis, final_response)
                steps.append({"step": f"verification_round_{i}", "output": verification})
                
                # Check if good enough
                coverage = verification.get("overall_coverage", 0.0)
                missing = verification.get("missing_aspects", [])
                critical = verification.get("critical_issues", [])
                
                is_perfect = coverage >= 0.95 and not critical and not missing
                
                if is_perfect or i >= self.max_refines:
                    break
                
                # Refine
                print(f"    -> Refining IARO response (Round {i+1})...")
                final_response = await self._refine_response(query, final_response, verification)
                steps.append({"step": f"refinement_round_{i+1}", "output": final_response[:500] + "..."})
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=final_response,
            intent_analysis=intent_analysis,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
            intermediate_steps=steps,
        )


# ========== Reflexion Solver ==========
class ReflexionSolver(BaseSolver):
    """
    Reflexion Solver - Verbal Reinforcement Learning
    
    基于 Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
    
    核心思想：通过语言反馈进行自我反思和改进
    流程：
    1. 初始生成
    2. 评估响应质量
    3. 生成反思 (verbal reflection)
    4. 基于反思改进响应
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant."""
    
    EVALUATE_PROMPT = """Evaluate the following response to the user's request.

User Request: {query}

Response:
{response}

Evaluate on these criteria:
1. Completeness: Does it address all parts of the request?
2. Correctness: Is the information/code accurate?
3. Quality: Is it well-structured and clear?

Output a score (1-10) and list specific issues found.

Format:
Score: X/10
Issues:
- [issue 1]
- [issue 2]
..."""

    REFLECT_PROMPT = """Based on the evaluation, reflect on how to improve the response.

User Request: {query}

Previous Response:
{response}

Evaluation:
{evaluation}

Generate a brief reflection on:
1. What went wrong or was missing
2. What specific improvements are needed
3. Key lessons for the improved response

Keep reflection concise (under 100 words)."""

    IMPROVE_PROMPT = """Improve your response based on the reflection.

User Request: {query}

Previous Response:
{response}

Reflection:
{reflection}

Now generate an improved response that addresses all the issues identified."""

    def __init__(self, client: Optional[AsyncLLMClient] = None, max_iterations: int = 1):
        super().__init__(client, name="Reflexion")
        self.max_iterations = max_iterations
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """Reflexion iterative improvement"""
        import time
        start = time.time()
        
        steps = []
        
        # Step 1: Initial generation
        response = await self.client.generate(
            prompt=query,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=1500,
        )
        steps.append({"step": "initial_generation", "output": response[:500]})
        
        # Reflexion loop
        for i in range(self.max_iterations):
            # Step 2: Evaluate
            eval_prompt = self.EVALUATE_PROMPT.format(query=query, response=response)
            evaluation = await self.client.generate(
                prompt=eval_prompt,
                temperature=0.3,
                max_tokens=500,
            )
            steps.append({"step": f"evaluation_{i+1}", "output": evaluation})
            
            # Check if score is high enough (extract score)
            try:
                score_line = [l for l in evaluation.split('\n') if 'score' in l.lower()][0]
                score = int(''.join(filter(str.isdigit, score_line.split('/')[0])))
                if score >= 9:
                    break
            except:
                pass
            
            # Step 3: Reflect
            reflect_prompt = self.REFLECT_PROMPT.format(
                query=query, response=response, evaluation=evaluation
            )
            reflection = await self.client.generate(
                prompt=reflect_prompt,
                temperature=0.5,
                max_tokens=200,
            )
            steps.append({"step": f"reflection_{i+1}", "output": reflection})
            
            # Step 4: Improve
            improve_prompt = self.IMPROVE_PROMPT.format(
                query=query, response=response, reflection=reflection
            )
            response = await self.client.generate(
                prompt=improve_prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=2000,
            )
            steps.append({"step": f"improved_response_{i+1}", "output": response[:500]})
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
            intermediate_steps=steps,
        )


# ========== Task Complexity Estimator ==========
def estimate_task_complexity(query: str) -> float:
    """
    估计任务复杂度 (0.0 - 1.0)
    
    高复杂度任务 (需要 IARO):
    - 涉及安全/生产环境的任务
    - 有场景描述的开放式任务
    - 涉及用户交互的任务
    
    低复杂度任务 (跳过 IARO):
    - 简单算法题 (MBPP 风格: "write a function to...")
    - 分类/提取任务
    - 简答题/选择题
    - 纯数学计算
    
    Returns:
        float: 0.0-1.0, 越高表示越需要 IARO
    """
    query_lower = query.lower()
    words = query_lower.split()
    word_count = len(words)
    
    # 高复杂度信号 (+) - 真正需要考虑隐式需求的场景
    high_complexity_signals = [
        'help me', 'i need', 'can you', 'please',
        'production', 'robust', 'secure', 'safe', 'scalable',
        'project', 'application', 'system', 'service', 'api',
        'user', 'client', 'customer', 'team',
        'deploy', 'maintain', 'handle errors',
    ]
    
    # 低复杂度信号 (-) - 不需要 IARO 的简单任务
    low_complexity_signals = [
        # 分类/提取
        'classify', 'categorize', 'sentiment', 'label',
        'extract', 'select', 'choose', 'pick', 'which',
        # 简答
        'what is', 'define', 'true or false', 'yes or no',
        'answer:', 'question:', 'options:', 'a)', 'b)', 'c)',
        # 简单编程题 (MBPP 风格)
        'write a function to', 'write a python function',
        'write function to', 'python function to',
        'find the', 'calculate the', 'compute the', 'count the',
        'check if', 'check whether', 'return true', 'return false',
        'given a list', 'given an array', 'given a string',
        'nth', 'maximum', 'minimum', 'sum of', 'product of',
        'sort', 'reverse', 'remove', 'replace',
    ]
    
    # 计算分数
    score = 0.5  # 基础分
    
    # 高复杂度信号加分
    for signal in high_complexity_signals:
        if signal in query_lower:
            score += 0.1
    
    # 低复杂度信号减分 (更强的减分)
    for signal in low_complexity_signals:
        if signal in query_lower:
            score -= 0.12
    
    # 长度因素
    if word_count > 80:
        score += 0.15  # 很长的描述通常有隐式需求
    elif word_count < 20:
        score -= 0.1   # 短描述通常是简单任务
    
    # 问号数量 (多问号可能是简单问答)
    question_marks = query.count('?')
    if question_marks > 2:
        score -= 0.1
    
    # 限制在 [0, 1]
    return max(0.0, min(1.0, score))


# ========== Adaptive IARO Solver ==========
class AdaptiveIAROSolver(IAROSolver):
    """
    自适应 IARO 求解器
    
    根据任务复杂度自动决定是否启用完整 IARO 流程:
    - 高复杂度任务: 使用完整 IARO (Intent Analysis + Generation)
    - 低复杂度任务: 退化为 Vanilla (直接回答)
    
    这解决了 IARO 在简单任务 (SST5, RACE, MBPP) 上的性能下降问题。
    """
    
    def __init__(
        self,
        client: Optional[AsyncLLMClient] = None,
        complexity_threshold: float = 0.4,
        **kwargs
    ):
        """
        Args:
            client: LLM 客户端
            complexity_threshold: 复杂度阈值，低于此值使用 Vanilla
        """
        super().__init__(client, **kwargs)
        self.name = "IARO-Adaptive"
        self.complexity_threshold = complexity_threshold
        self.vanilla_solver = VanillaSolver(client)
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        """自适应求解"""
        complexity = estimate_task_complexity(query)
        
        if complexity < self.complexity_threshold:
            # 简单任务：使用 Vanilla
            response = await self.vanilla_solver.solve(query, **kwargs)
            response.method = f"{self.name} (→Vanilla, complexity={complexity:.2f})"
            return response
        else:
            # 复杂任务：使用完整 IARO
            response = await super().solve(query, **kwargs)
            response.method = f"{self.name} (→IARO, complexity={complexity:.2f})"
            return response


# ========== IARO-Recursive Solver (with Verification Loop) ==========
class IARORecursiveSolver(IAROSolver):
    """
    IARO-Recursive: 带验证循环的完整 IARO
    
    与 IARO-Base (2-step) 的区别:
    - 启用 Verification 模块检查响应覆盖率
    - 支持 Refinement Loop 进行迭代改进
    - 支持 Intent Expansion 动态发现新需求
    
    适用场景: 极难的、高风险的任务（如医疗、金融、安全关键代码）
    代价: 更多 API 调用 (4+ calls vs 2 calls)
    """
    
    def __init__(
        self,
        client: Optional[AsyncLLMClient] = None,
        max_refines: int = 1,
        enable_intent_expansion: bool = True,
    ):
        super().__init__(
            client=client,
            enable_verification=True,
            max_refines=max_refines,
            enable_intent_expansion=enable_intent_expansion,
            name="IARO-Recursive",
        )


# ========== IARO-Selective Solver (Conditional Refinement) ==========
class IAROSelectiveSolver(IAROSolver):
    """
    IARO-Selective: 条件性触发 Refinement 的改进版本
    
    解决 IARO-Recursive 的 Over-Expansion 问题:
    - 只在覆盖率低于阈值时触发 Refinement
    - 只使用 Blocker 类型的新需求
    - 限制新需求数量避免响应膨胀
    
    适用场景: 需要 Intent Expansion 但要避免过度修改的场景
    """
    
    def __init__(
        self,
        client: Optional[AsyncLLMClient] = None,
        max_refines: int = 1,
        coverage_threshold: float = 0.7,  # 只在覆盖率 < 70% 时触发
        max_new_needs: int = 2,           # 最多添加 2 个新需求
    ):
        super().__init__(
            client=client,
            enable_verification=True,
            max_refines=max_refines,
            enable_intent_expansion=True,
            name="IARO-Selective",
        )
        self.coverage_threshold = coverage_threshold
        self.max_new_needs = max_new_needs
    
    async def _refine_response(
        self,
        query: str,
        response: str,
        verification: Dict[str, Any],
    ) -> str:
        """Selective Refinement: 只在必要时修改，只用 Blocker 需求"""
        # 获取覆盖率，默认 0.5 而不是 1.0（保守估计）
        coverage = verification.get("overall_coverage", 0.5)
        
        # 获取新发现的需求
        new_needs = verification.get("newly_identified_needs", [])
        critical = verification.get("critical_issues", [])
        missing = verification.get("missing_aspects", [])
        
        # 调试输出
        print(f"    [Selective] coverage={coverage:.2f}, new_needs={len(new_needs)}, critical={len(critical)}")
        
        # 核心决策逻辑：
        # 1. 如果有 critical issues，必须 refine
        # 2. 如果覆盖率低，需要 refine
        # 3. 如果有 blocker 类型的新需求，考虑 refine
        
        has_critical = len(critical) > 0
        low_coverage = coverage < self.coverage_threshold
        
        # 筛选 Blocker 需求
        blocker_needs = []
        for need in new_needs:
            if isinstance(need, dict):
                if need.get("integration_type", "").lower() == "blocker":
                    blocker_needs.append(need)
        blocker_needs = blocker_needs[:self.max_new_needs]
        
        # 决策：如果没有紧急问题且覆盖率足够，跳过
        if not has_critical and not low_coverage and not blocker_needs:
            print(f"    -> Skipping refinement (coverage={coverage:.2f} >= {self.coverage_threshold}, no critical issues)")
            return response
        
        # 如果只是覆盖率高但有 blocker needs，仍然 refine
        if coverage >= self.coverage_threshold and not has_critical and blocker_needs:
            print(f"    -> Selective refinement: adding {len(blocker_needs)} blocker needs")
        
        print(f"    -> Proceeding with refinement (critical={len(critical)}, blocker_needs={len(blocker_needs)}, missing={len(missing)})")
        
        # 构建精简的反馈
        def to_str(item):
            if isinstance(item, str):
                return item
            elif isinstance(item, dict):
                return item.get("description", item.get("issue", str(item)))
            return str(item)
        
        feedback_lines = []
        if critical:
            feedback_lines.append(f"CRITICAL ISSUES (must fix): {'; '.join(to_str(c) for c in critical[:2])}")
        
        if blocker_needs:
            feedback_lines.append(f"BLOCKER NEEDS (add briefly): {'; '.join(to_str(n) for n in blocker_needs)}")
        
        if missing:
            feedback_lines.append(f"MISSING: {'; '.join(to_str(m) for m in missing[:2])}")
        
        feedback = "\n".join(feedback_lines)
        
        # 使用更保守的 Refinement Prompt
        prompt = f"""Minimally improve the response. Make SMALL, TARGETED changes only.

## Original Request:
{query}

## Current Response (keep most of this):
{response}

## Issues to Address (be brief):
{feedback}

## Guidelines:
- Make MINIMAL changes - do NOT rewrite the entire response
- Add missing critical elements BRIEFLY (1-2 sentences max per issue)
- Do NOT add extensive new sections
- Keep the response length similar to the original

Output the improved response:"""
        
        refined = await self.client.generate(
            prompt=prompt,
            system_prompt="You are an editor making minimal, targeted improvements.",
            temperature=0.3,
            max_tokens=2000,
        )
        return refined


# ========== IARO-Lite Solver (for MT-Bench) ==========
class IAROLiteSolver(BaseSolver):
    """
    轻量级 IARO - 静默意图注入
    
    适用于多轮对话等需要保持输出格式自然的场景 (如 MT-Bench)。
    
    工作方式:
    1. 静默分析隐式需求
    2. 将约束注入 system prompt 而非用户 prompt
    3. 使用标准生成，保持输出格式自然
    """
    
    SILENT_ANALYSIS_PROMPT = """Briefly identify any implicit requirements in this request that aren't explicitly stated.
Focus only on critical safety or correctness concerns.

Request: {query}

Output a brief comma-separated list of implicit needs, or "none" if the request is straightforward.
Keep it under 50 words."""

    def __init__(self, client: Optional[AsyncLLMClient] = None):
        super().__init__(client, name="IARO-Lite")
    
    async def solve(self, query: str, **kwargs) -> SolverResponse:
        import time
        start = time.time()
        
        # 1. 静默分析 (快速)
        implicit_needs = await self.client.generate(
            prompt=self.SILENT_ANALYSIS_PROMPT.format(query=query),
            temperature=0.3,
            max_tokens=100,
        )
        
        # 2. 注入 system prompt
        if implicit_needs.lower().strip() != "none":
            enhanced_system = f"""You are a helpful assistant.

[Internal guidance - do not mention these explicitly: Consider {implicit_needs}]

Respond naturally and conversationally."""
        else:
            enhanced_system = "You are a helpful assistant."
        
        # 3. 标准生成
        response = await self.client.generate(
            prompt=query,
            system_prompt=enhanced_system,
            temperature=0.7,
            max_tokens=2048,
        )
        
        latency = (time.time() - start) * 1000
        
        return SolverResponse(
            method=self.name,
            query=query,
            response=response,
            latency_ms=latency,
            model=self.client.model,
            timestamp=datetime.now().isoformat(),
        )


# ========== Solver Factory ==========
class SolverFactory:
    """求解器工厂"""
    
    SOLVERS = {
        "vanilla": VanillaSolver,
        "cot": CoTSolver,
        "fewshot_cot": FewShotCoTSolver,
        "rar": RaRSolver,              # New: Rephrase-and-Respond (ICLR 2024)
        "critic": CRITICSolver,          # New: CRITIC (ICLR 2024)
        "self_refine": SelfRefineSolver,
        "reflexion": ReflexionSolver,
        "iaro": IAROSolver,              # 2-step base (Analyze -> Generate)
        "iaro_recursive": IARORecursiveSolver,  # Full loop with verification
        "iaro_selective": IAROSelectiveSolver,  # Conditional refinement (improved)
        "iaro_adaptive": AdaptiveIAROSolver,
        "iaro_lite": IAROLiteSolver,
    }
    
    @classmethod
    def create(
        cls,
        method: str,
        client: Optional[AsyncLLMClient] = None,
        **kwargs
    ) -> BaseSolver:
        """
        创建求解器
        
        Args:
            method: 方法名称
            client: LLM客户端
            **kwargs: 额外参数
        
        Returns:
            BaseSolver实例
        """
        method = method.lower()
        if method not in cls.SOLVERS:
            raise ValueError(f"Unknown method: {method}. Available: {list(cls.SOLVERS.keys())}")
        
        return cls.SOLVERS[method](client=client, **kwargs)
    
    @classmethod
    def create_all(
        cls,
        client: Optional[AsyncLLMClient] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, BaseSolver]:
        """
        创建所有求解器
        
        Args:
            client: LLM客户端
            methods: 要创建的方法列表，None则创建全部
        
        Returns:
            方法名到求解器的映射
        """
        methods = methods or list(cls.SOLVERS.keys())
        return {m: cls.create(m, client) for m in methods}


# ========== 测试代码 ==========
async def test_solvers():
    """测试所有求解器"""
    print("=" * 60)
    print("Testing Solvers")
    print("=" * 60)
    
    # 测试查询
    test_query = "Write a Python script to process CSV files and extract user emails"
    
    print(f"\nTest Query: {test_query}\n")
    
    # 创建所有求解器
    solvers = SolverFactory.create_all()
    
    for name, solver in solvers.items():
        print(f"\n{'='*40}")
        print(f"Testing: {name}")
        print("=" * 40)
        
        response = await solver.solve(test_query)
        
        print(f"Response Preview: {response.response[:300]}...")
        print(f"Latency: {response.latency_ms:.2f}ms")
        
        if response.intent_analysis:
            print(f"Implicit Needs Identified: {len(response.intent_analysis.get('implicit_needs', []))}")


if __name__ == "__main__":
    asyncio.run(test_solvers())
