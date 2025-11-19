"""
Coordinator Agent - The Brain of the Multi-Agent System

This agent autonomously decides what to do next to achieve the goal (top 20% ranking).
It observes the current state, reasons about what's needed, and calls appropriate specialist agents.

This is TRUE AGENCY - no hardcoded workflow, the AI decides everything.
"""
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from ...utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseLLMAgent):
    """
    Autonomous coordinator that decides workflow dynamically.

    The coordinator is the "brain" - it observes the current state and autonomously
    decides what specialist agent to call next to achieve the goal.

    Philosophy: NO HARDCODED WORKFLOW. The AI decides everything based on:
    - Current state
    - Goal (top 20% ranking)
    - Available specialist agents (tools)
    - Past actions and results
    """

    def __init__(self):
        # Load system prompt
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "coordinator_agent.txt"
        system_prompt = prompt_file.read_text() if prompt_file.exists() else self._get_default_system_prompt()

        super().__init__(
            name="CoordinatorAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,  # Some creativity for strategy, but not too much
            system_prompt=system_prompt
        )

        # Available specialist agents (tools)
        self.available_actions = {
            "collect_data": "Download competition data from Kaggle",
            "understand_problem": "Read and understand competition requirements",
            "analyze_data": "Analyze data and identify preprocessing needs",
            "preprocess_data": "Generate and execute preprocessing code",
            "plan_strategy": "Create model training strategy and plan",
            "engineer_features": "Generate and execute feature engineering code",
            "train_model": "Train models according to plan",
            "submit_predictions": "Generate predictions and submit to Kaggle",
            "evaluate_results": "Check leaderboard and diagnose performance",
            "optimize_strategy": "Analyze failures and create improvement plan",
            "done": "Goal achieved or no more improvements possible"
        }

    def _get_default_system_prompt(self) -> str:
        return """You are an autonomous AI coordinator managing a Kaggle competition workflow.

Your goal: Achieve top 20% ranking in ANY Kaggle competition.

Your role: Observe the current state and autonomously decide what to do next.

You have access to specialist agents (tools):
- collect_data: Download competition files
- understand_problem: Understand competition requirements
- analyze_data: Analyze data characteristics and needs
- preprocess_data: Clean and prepare data
- plan_strategy: Decide which models and strategies to use
- engineer_features: Create new features
- train_model: Train machine learning models
- submit_predictions: Submit to Kaggle
- evaluate_results: Check performance and diagnose issues
- optimize_strategy: Improve based on results
- done: Declare completion

Key principles:
1. You decide the workflow - no fixed sequence
2. You can skip steps if not needed
3. You can repeat steps if needed
4. You learn from previous actions
5. You adapt strategy based on results
6. You decide when you're done

Be strategic, adaptive, and goal-oriented."""

    async def coordinate(
        self,
        goal: str,
        current_state: Dict[str, Any],
        action_history: list,
        max_actions: int = 50
    ) -> Dict[str, Any]:
        """
        Autonomously coordinate the workflow to achieve the goal.

        Args:
            goal: The objective (e.g., "Achieve top 20% in titanic competition")
            current_state: Current context/state of the system
            action_history: List of previous actions taken
            max_actions: Maximum number of actions to prevent infinite loops

        Returns:
            Final state after achieving goal or exhausting actions
        """
        logger.info("🧠 Coordinator Agent starting autonomous workflow...")
        logger.info(f"📍 Goal: {goal}")
        logger.info(f"📊 Initial state: {list(current_state.keys())}")

        state = current_state.copy()
        actions_taken = 0

        while actions_taken < max_actions:
            actions_taken += 1

            logger.info(f"\n{'='*70}")
            logger.info(f"🤔 ACTION {actions_taken}/{max_actions}: Coordinator deciding next move...")
            logger.info(f"{'='*70}")

            # Coordinator autonomously decides next action
            decision = await self._decide_next_action(goal, state, action_history)

            action = decision.get("action")
            reasoning = decision.get("reasoning", "No reasoning provided")

            logger.info(f"🎯 Decision: {action}")
            logger.info(f"💭 Reasoning: {reasoning}")

            # Check if done
            if action == "done":
                logger.info("✅ Coordinator declares: GOAL ACHIEVED or no further improvements possible")
                state["coordinator_final_decision"] = decision
                break

            # Record the decision
            action_history.append({
                "action_number": actions_taken,
                "action": action,
                "reasoning": reasoning,
                "state_before": self._summarize_state(state)
            })

            # This is the key: coordinator calls specialist agent
            # The actual execution will be handled by the orchestrator
            # Coordinator just decides WHAT to do, orchestrator executes it
            state["next_action"] = action
            state["action_reasoning"] = reasoning

            # Return to orchestrator for execution
            # Orchestrator will execute the action and call coordinator again
            return {
                "action": action,
                "reasoning": reasoning,
                "continue": True,
                "state": state,
                "action_history": action_history
            }

        # Max actions reached
        logger.warning(f"⚠️ Reached maximum actions ({max_actions}) - stopping")
        state["coordinator_final_decision"] = {
            "action": "done",
            "reasoning": f"Exhausted maximum actions ({max_actions})"
        }

        return {
            "action": "done",
            "reasoning": f"Maximum actions reached ({max_actions})",
            "continue": False,
            "state": state,
            "action_history": action_history
        }

    async def _decide_next_action(
        self,
        goal: str,
        state: Dict[str, Any],
        action_history: list
    ) -> Dict[str, str]:
        """
        The core decision-making function - AI autonomously decides what to do next.

        This is TRUE AGENCY - no hardcoded logic, pure AI reasoning.
        """

        # Build comprehensive prompt with current state
        prompt = self._build_decision_prompt(goal, state, action_history)

        # Get AI decision
        response = generate_ai_response(self.model, prompt)

        # Parse decision
        try:
            decision = self._parse_decision(response)
            return decision
        except Exception as e:
            logger.error(f"❌ Failed to parse coordinator decision: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            # Fallback decision
            return {
                "action": "done",
                "reasoning": f"Failed to parse decision: {e}"
            }

    def _build_decision_prompt(
        self,
        goal: str,
        state: Dict[str, Any],
        action_history: list
    ) -> str:
        """Build prompt for coordinator to make decision."""

        # Summarize current state
        state_summary = self._get_detailed_state_summary(state)

        # Summarize action history
        history_summary = self._get_action_history_summary(action_history)

        prompt = f"""You are the Coordinator Agent deciding the next action to achieve the goal.

## GOAL
{goal}

## CURRENT STATE
{state_summary}

## ACTION HISTORY
{history_summary}

## AVAILABLE ACTIONS
{json.dumps(self.available_actions, indent=2)}

## YOUR TASK

Analyze the current state and decide what to do NEXT to achieve the goal.

**Decision Rules:**
1. If data not collected → collect_data
2. If problem not understood → understand_problem
3. If data not analyzed → analyze_data
4. If preprocessing needed and not done → preprocess_data
5. If no strategy planned → plan_strategy
6. If features needed → engineer_features
7. If model not trained → train_model
8. If predictions not submitted → submit_predictions
9. If results not evaluated → evaluate_results
10. If need improvement → optimize_strategy (then loop)
11. If goal achieved → done
12. If stuck or no improvements possible → done

**Important:**
- Skip steps that aren't needed (AI decides based on data analysis)
- You can revisit steps (e.g., retrain with different strategy)
- Be adaptive - change strategy based on results
- Declare "done" when goal achieved or no more improvements possible

Return ONLY valid JSON:

{{
  "action": "one of the available actions",
  "reasoning": "brief explanation of why this action is needed now (2-3 sentences)",
  "confidence": "high|medium|low"
}}

**Examples:**

Example 1 (First iteration):
{{
  "action": "collect_data",
  "reasoning": "No data has been collected yet. This is the first step - we need to download the competition files before we can do anything else.",
  "confidence": "high"
}}

Example 2 (After poor first submission):
{{
  "action": "optimize_strategy",
  "reasoning": "Current model achieved 0.72 accuracy but goal is top 20% (requires ~0.80). CV score of 0.78 shows overfitting. Need to analyze and improve strategy.",
  "confidence": "high"
}}

Example 3 (Goal achieved):
{{
  "action": "done",
  "reasoning": "Achieved 18th percentile ranking, exceeding the 20% goal. Model is performing well with stable CV and LB scores.",
  "confidence": "high"
}}

Remember: Return ONLY the JSON object, no markdown, no explanations outside JSON.
"""

        return prompt

    def _get_detailed_state_summary(self, state: Dict[str, Any]) -> str:
        """Create detailed summary of current state for decision-making."""

        summary_parts = []

        # Competition info
        summary_parts.append(f"**Competition:** {state.get('competition_name', 'Unknown')}")
        summary_parts.append(f"**Target Percentile:** {state.get('target_percentile', 0.20) * 100}%")
        summary_parts.append(f"**Iteration:** {state.get('iteration', 0)}")

        # Phase completion status
        summary_parts.append("\n**Phase Status:**")
        summary_parts.append(f"- Data collected: {'✅' if state.get('data_path') else '❌'}")
        summary_parts.append(f"- Problem understood: {'✅' if state.get('problem_understanding') else '❌'}")
        summary_parts.append(f"- Data analyzed: {'✅' if state.get('data_analysis') else '❌'}")
        summary_parts.append(f"- Data preprocessed: {'✅' if state.get('clean_data_path') else '❌' if state.get('needs_preprocessing') else '⏭️ Not needed'}")
        summary_parts.append(f"- Strategy planned: {'✅' if state.get('execution_plan') else '❌'}")
        summary_parts.append(f"- Features engineered: {'✅' if state.get('featured_data_path') else '❌' if state.get('needs_feature_engineering') else '⏭️ Not needed'}")
        summary_parts.append(f"- Model trained: {'✅' if state.get('model_path') else '❌'}")
        summary_parts.append(f"- Predictions submitted: {'✅' if state.get('submission_id') else '❌'}")
        summary_parts.append(f"- Results evaluated: {'✅' if state.get('current_percentile') is not None else '❌'}")

        # Performance metrics (if available)
        if state.get('cv_score') or state.get('leaderboard_score'):
            summary_parts.append("\n**Performance:**")
            if state.get('cv_score'):
                summary_parts.append(f"- CV Score: {state.get('cv_score'):.4f}")
            if state.get('leaderboard_score'):
                summary_parts.append(f"- Leaderboard Score: {state.get('leaderboard_score'):.4f}")
            if state.get('current_percentile'):
                summary_parts.append(f"- Current Percentile: {state.get('current_percentile') * 100:.1f}%")
                summary_parts.append(f"- Gap to Target: {(state.get('current_percentile', 1) - state.get('target_percentile', 0.2)) * 100:.1f}%")
            if state.get('meets_target'):
                summary_parts.append("- **TARGET ACHIEVED! ✅**")

        # Key insights from data analysis
        if state.get('data_analysis', {}).get('key_insights'):
            summary_parts.append("\n**Key Data Insights:**")
            for insight in state.get('data_analysis', {}).get('key_insights', [])[:3]:
                summary_parts.append(f"- {insight}")

        return "\n".join(summary_parts)

    def _get_action_history_summary(self, action_history: list) -> str:
        """Summarize previous actions taken."""
        if not action_history:
            return "No actions taken yet (starting fresh)"

        summary = f"**Total actions taken:** {len(action_history)}\n\n"

        # Show last 5 actions
        recent_actions = action_history[-5:]
        summary += "**Recent actions:**\n"
        for entry in recent_actions:
            summary += f"{entry['action_number']}. {entry['action']}: {entry['reasoning']}\n"

        return summary

    def _summarize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create lightweight summary of state for history."""
        return {
            "has_data": state.get('data_path') is not None,
            "has_model": state.get('model_path') is not None,
            "submitted": state.get('submission_id') is not None,
            "percentile": state.get('current_percentile'),
            "meets_target": state.get('meets_target')
        }

    def _parse_decision(self, response: str) -> Dict[str, str]:
        """Parse coordinator's decision from response."""
        response = response.strip()

        # Remove markdown if present
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.rfind("```")
            if start > 6 and end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.rfind("```")
            if start > 2 and end > start:
                response = response[start:end].strip()

        # Parse JSON
        decision = json.loads(response)

        # Validate
        if "action" not in decision:
            raise ValueError("Decision missing 'action' field")

        if decision["action"] not in self.available_actions and decision["action"] != "done":
            raise ValueError(f"Invalid action: {decision['action']}")

        return decision