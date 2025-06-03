#!/usr/bin/env python3
"""
Phase 2: LLM-Based Code Complexity Analysis
Uses OpenAI API to provide semantic complexity analysis of the top complex functions
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from analysis.llm_prompt import create_analysis_prompt
from openai import OpenAI


@dataclass
class LLMComplexityMetrics:
    """Container for LLM-based complexity analysis"""

    semantic_complexity: int = 0  # 1-10 scale
    cognitive_load: int = 0  # 1-10 scale
    maintainability: int = 0  # 1-10 scale
    documentation_quality: int = 0  # 1-10 scale
    refactoring_urgency: int = 0  # 1-10 scale
    explanation: str = ""
    suggestions: List[str] = None
    business_description: str = ""  # New field
    developer_description: str = ""  # New field
    llm_score: float = 0.0

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class LLMComplexityAnalyzer:
    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM analyzer

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens_per_request = 4000  # Adjust based on your model

        # Language-specific patterns for dependency extraction
        self.dependency_patterns = {
            "java": r"\b(\w+)\s*\(",
            "kotlin": r"\b(\w+)\s*\(",
            "typescript": r"\b(\w+)\s*\(",
            "javascript": r"\b(\w+)\s*\(",
            "groovy": r"\b(\w+)\s*\(",
            "python": r"\b(\w+)\s*\(",
            "csharp": r"\b(\w+)\s*\(",
            "cpp": r"\b(\w+)\s*\(",
            "go": r"\b(\w+)\s*\(",
        }

    def get_file_path(self, function_data: Dict[str, Any]) -> str:
        """Extract file path from function data, handling both file_path and file_url"""
        if "file_path" in function_data:
            return function_data["file_path"]
        elif "file_url" in function_data:
            # Convert file URL to path
            file_url = function_data["file_url"]
            if file_url.startswith("file:///"):
                # Remove file:/// prefix and convert to normal path
                return file_url[8:].replace("/", os.sep)
            return file_url
        else:
            return "unknown"

    def extract_function_calls(self, code: str, language: str) -> List[str]:
        """Extract function calls from code"""
        if language not in self.dependency_patterns:
            return []

        pattern = self.dependency_patterns[language]
        matches = re.findall(pattern, code)

        # Filter out common keywords and built-ins
        keywords = {
            "if",
            "for",
            "while",
            "switch",
            "try",
            "catch",
            "return",
            "new",
            "class",
            "function",
            "var",
            "let",
            "const",
            "def",
            "print",
            "console",
            "log",
            "toString",
            "length",
            "size",
        }

        return [match for match in set(matches) if match.lower() not in keywords]

    def find_related_functions(
        self, target_function: Dict[str, Any], all_functions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find functions that are called by the target function"""
        if not target_function.get("function_content"):
            return []

        # Extract function calls from target function
        called_functions = self.extract_function_calls(
            target_function["function_content"], target_function["language"]
        )

        # Get file paths
        target_file_path = self.get_file_path(target_function)

        # Find matching functions in the codebase
        related = []
        for func in all_functions:
            if func["function_name"] in called_functions:
                func_file_path = self.get_file_path(func)
                # Limit to same file or same package for relevance
                if func_file_path == target_file_path or os.path.dirname(
                    func_file_path
                ) == os.path.dirname(target_file_path):
                    related.append(func)

        # Limit to top 5 most relevant dependencies
        return related[:5]

    def build_analysis_context(
        self, target_function: Dict[str, Any], related_functions: List[Dict[str, Any]]
    ) -> str:
        """Build comprehensive context for LLM analysis"""

        context_parts = []

        # Add target function
        context_parts.append("=== TARGET FUNCTION FOR ANALYSIS ===")
        context_parts.append(f"Function: {target_function['function_name']}")
        context_parts.append(f"File: {self.get_file_path(target_function)}")
        context_parts.append(f"Language: {target_function['language']}")
        context_parts.append(
            f"Lines: {target_function['start_line']}-{target_function['end_line']}"
        )
        context_parts.append("\nPhase 1 Structural Complexity Metrics:")
        for metric, value in target_function["rule_analysis"].items():
            context_parts.append(f"  - {metric}: {value}")

        context_parts.append(
            f"\nStructural Complexity Score: {target_function['rule_analysis']['rule_score']:.2f}"
        )

        # Add function code if available
        if target_function.get("function_content"):
            context_parts.append("\n=== FUNCTION CODE ===")
            context_parts.append(target_function["function_content"])
        else:
            context_parts.append("\n=== FUNCTION CODE ===")
            context_parts.append(
                "(Function content not available - analyzing based on metrics only)"
            )

        # Add related functions for context
        if related_functions:
            context_parts.append("\n=== RELATED FUNCTIONS (Dependencies) ===")
            for i, func in enumerate(related_functions, 1):
                context_parts.append(
                    f"\n--- Related Function {i}: {func['function_name']} ---"
                )
                if func.get("function_content"):
                    context_parts.append(func["function_content"])
                else:
                    context_parts.append("(Content not available)")

        return "\n".join(context_parts)

    def call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Make API call to OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code complexity analyzer. Always respond with valid JSON in the exact format requested. Do not include any text before or after the JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens_per_request,
                temperature=0.1,  # Low temperature for consistent analysis
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response if it contains extra text
            content = self._extract_json_from_response(content)

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {content}")
            return self._create_fallback_response()

        except Exception as e:
            print(f"API call error: {e}")
            return self._create_fallback_response()

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from response that might contain extra text"""
        # Try to find JSON block in the response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, return the content as-is and let it fail gracefully
        return content

    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create fallback response if API fails"""
        return {
            "semantic_complexity": 5,
            "cognitive_load": 5,
            "maintainability": 5,
            "documentation_quality": 5,
            "refactoring_urgency": 5,
            "explanation": "API analysis failed - using fallback scores",
            "business_description": "Function purpose could not be determined due to API failure",
            "developer_description": "Technical analysis unavailable due to API failure",
            "suggestions": [
                "Review this function manually",
                "Consider refactoring",
                "Add documentation",
            ],
        }

    def calculate_final_score(
        self, llm_metrics: LLMComplexityMetrics, structural_score: float
    ) -> float:
        """Combine LLM analysis with structural metrics for final score"""

        # Weight LLM metrics (normalize to 1-10 scale)
        llm_metrics.llm_score = (
            llm_metrics.semantic_complexity * 3.0
            + llm_metrics.cognitive_load * 2.5
            + llm_metrics.maintainability * 2.0
            + llm_metrics.documentation_quality * 1.5
            + llm_metrics.refactoring_urgency * 2.0
        ) / 11.0  # Normalize to 1-10 scale

        # Normalize structural score to 1-10 scale
        # Assuming structural scores can be quite high, we need to map them appropriately
        # You may need to adjust this based on your typical structural score ranges
        normalized_structural = min(structural_score / 10.0, 10.0)

        # Alternative normalization if structural scores are typically much higher:
        # normalized_structural = min(structural_score / 50.0 * 10.0, 10.0)

        # Combine scores (60% LLM, 40% structural)
        final_score = (llm_metrics.llm_score * 0.6) + (normalized_structural * 0.4)

        return final_score

    def analyze_function(
        self, target_function: Dict[str, Any], all_functions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a single function with LLM"""

        print(f"Analyzing function: {target_function['function_name']}")
        related_functions = self.find_related_functions(target_function, all_functions)
        context = self.build_analysis_context(target_function, related_functions)
        prompt = create_analysis_prompt(context)
        llm_response = self.call_openai_api(prompt)
        llm_metrics = LLMComplexityMetrics(
            semantic_complexity=llm_response.get("semantic_complexity", 5),
            cognitive_load=llm_response.get("cognitive_load", 5),
            maintainability=llm_response.get("maintainability", 5),
            documentation_quality=llm_response.get("documentation_quality", 5),
            refactoring_urgency=llm_response.get("refactoring_urgency", 5),
            explanation=llm_response.get("explanation", ""),
            business_description=llm_response.get("business_description", ""),
            developer_description=llm_response.get("developer_description", ""),
            suggestions=llm_response.get("suggestions", []),
        )

        final_score = self.calculate_final_score(
            llm_metrics, target_function["rule_analysis"]["rule_score"]
        )

        enhanced_function = target_function.copy()
        enhanced_function.update(
            {
                "llm_analysis": {
                    "semantic_complexity": llm_metrics.semantic_complexity,
                    "cognitive_load": llm_metrics.cognitive_load,
                    "maintainability": llm_metrics.maintainability,
                    "documentation_quality": llm_metrics.documentation_quality,
                    "refactoring_urgency": llm_metrics.refactoring_urgency,
                    "explanation": llm_metrics.explanation,
                    "business_description": llm_metrics.business_description,
                    "developer_description": llm_metrics.developer_description,
                    "suggestions": llm_metrics.suggestions,
                    "llm_score": llm_metrics.llm_score,
                },
                "combined_complexity_score": final_score,
            }
        )

        return enhanced_function

    def analyze_top_functions(
        self, complex_functions_file: str, top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Analyze top N most complex functions with LLM"""

        # Load Phase 1 results
        with open(complex_functions_file, "r") as f:
            all_functions = json.load(f)

        top_functions = all_functions[:top_n]

        print(f"Starting LLM analysis of top {len(top_functions)} functions...")
        enhanced_results = []
        for i, func in enumerate(top_functions, 1):
            print(f"Progress: {i}/{len(top_functions)}")

            try:
                enhanced_func = self.analyze_function(func, all_functions)
                enhanced_results.append(enhanced_func)
            except Exception as e:
                print(f"Error analyzing {func['function_name']}: {e}")
                # Add original function with error marker
                func["llm_analysis"] = {"error": str(e)}
                enhanced_results.append(func)

        # Re-sort by combined score
        enhanced_results.sort(
            key=lambda x: x.get("combined_complexity_score", 0), reverse=True
        )

        return enhanced_results


def main():
    """Main execution function for Phase 2"""

    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")  # Set your API key as environment variable
    if not API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return

    MODEL = "gpt-3.5-turbo"  # or "gpt-4" or "gpt-4-turbo" or "gpt-3.5-turbo"
    INPUT_FILE = "./complex_functions.json"  # Output from Phase 1
    OUTPUT_FILE = "./llm_analyzed_functions.json"
    TOP_N = 8  # Number of functions to analyze
    # TOP_N = 20

    analyzer = LLMComplexityAnalyzer(API_KEY, MODEL)
    results = analyzer.analyze_top_functions(INPUT_FILE, TOP_N)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 80}")
    print(f"LLM ANALYSIS COMPLETE - Top {len(results)} Functions")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
