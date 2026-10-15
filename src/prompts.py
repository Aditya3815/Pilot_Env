"""Prompts used by the pilot. Everything here lands in the candidate url4 strings."""

COVERAGE_PROMPT = (
    "You are given a user request and several candidate answers written independently. "
    "Produce one final answer to the original request. "
    "First identify every explicit instruction and constraint in the request (format, length, "
    "keywords, case, structure, forbidden content). "
    "Keep content the candidates agree on; where they conflict, choose what is best supported "
    "and most compliant. "
    "Add correct details that only one candidate included if they help satisfy the request. "
    "The final answer must satisfy every constraint in the original request exactly. "
    "Output only the final answer, with no commentary about the candidates."
)

JUDGE_RUBRIC = "Pick the answer that best follows every instruction in the request."
