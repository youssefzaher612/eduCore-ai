
from pydantic import ValidationError

from educore.structured_output import StructuredOutputError, parse_json_object
from educore.teacher_ai.core.llm_client import LLMTruncatedError, llm_client

from .schemas import CommunicationRequest, CommunicationResult


SYSTEM_PROMPT = """
You are an experienced school communication assistant.

Your task is to write clear, respectful, professional messages
for teachers communicating with parents, students, or school staff.

Rules:
- Use only the information provided by the user.
- Never invent grades, achievements, incidents, or student details.
- Keep the message constructive, specific, and supportive.
- Adapt the tone to the requested tone.
- Avoid harsh, judgmental, or embarrassing language.
- Return valid JSON only.
- Do not include Markdown code fences.

Required JSON structure:
{
    "subject_line": "Message subject",
    "message": "Complete message",
    "tone": "professional",
    "language": "English"
}
"""


def generate_communication(
    request: CommunicationRequest,
) -> CommunicationResult:

    if not isinstance(request, CommunicationRequest):
        request = CommunicationRequest.model_validate(request)

    user_prompt = f"""
Recipient type: {request.recipient_type}
Recipient name: {request.recipient_name}
Student name: {request.student_name}
Subject: {request.subject}
Grade level: {request.grade_level}
Purpose: {request.purpose}
Requested tone: {request.tone}
Additional context: {request.additional_context or "None"}

Write an appropriate message for this situation.

Return JSON only.
"""

    for attempt in range(2):
        try:
            response = llm_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2000,
            )
            data = parse_json_object(response)
            # Recipient details come from the validated request, not model output.
            return CommunicationResult.model_validate({
                **data,
                "recipient_type": request.recipient_type,
                "recipient_name": request.recipient_name,
                "tone": data.get("tone") or request.tone,
                "language": data.get("language") or "English",
            })
        except LLMTruncatedError as exc:
            if attempt:
                raise ValueError("The communication response was incomplete after one retry.") from exc
            user_prompt += "\n\nRETRY: Return a complete JSON object with concise, non-empty subject_line and message fields."
        except (StructuredOutputError, TypeError, ValueError, ValidationError) as exc:
            if attempt:
                raise ValueError(
                    "The AI returned an invalid communication message; expected non-empty "
                    "subject_line and message fields."
                ) from exc
            user_prompt += "\n\nCORRECTION: Return one valid JSON object with non-empty subject_line and message fields."

    raise ValueError("The AI could not create a valid communication message.")
