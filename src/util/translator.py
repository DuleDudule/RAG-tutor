import json
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage


class TranslationOutput(BaseModel):
    """Structured output schema for translation results"""

    original_query: str = Field(
        description="The original query in the source language (Serbian)"
    )
    translated_query: str = Field(
        description="The translated query in English, optimized for semantic search"
    )
    translation_confidence: float = Field(
        description="Confidence score of the translation (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    


def translate_serbian_to_english(
    query: str,
    llm: BaseChatModel | None = None
) -> TranslationOutput:
    """
    Translate a Serbian query to English using an LLM with structured output.

    Args:
        query: The users query in Serbian.
        llm: The language model to use for translation. If None, uses the default from env_check.get_llm()

    Returns:
        TranslationOutput: Structured result containing the original query, translated query and confidence score

    """

    from src.util.env_check import get_llm_model as get_default_llm

    model = llm if llm is not None else get_default_llm()

    system_prompt = """
        You are an expert translator specializing in Serbian to English translation.
        Your task is to translate technical/academic queries from Serbian to English.

        Guidelines:
        1. Preserve the meaning and intent of the original query exactly
        2. Maintain any technical terms, variable names, or code snippets as-is
        3. Optimize the translation for semantic search - use clear, natural English
        4. Keep the translation concise while preserving all key information
        5. If the query contains mixed languages, translate only the Serbian portions

        Return your response strictly following the TranslationOutput schema.
        """
    
    human_message = HumanMessage(
        content=f"Translate this Serbian query to English: {query}"
    )

    response = model.with_structured_output(TranslationOutput).invoke(
        [HumanMessage(content=system_prompt), human_message]
    )

    return response


def translate_query(
    query: str,
    source_language: str = "sr",
    target_language: str = "en",
    llm: BaseChatModel | None = None
) -> dict[str, Any]:
    """
    Generic translation function supporting multiple language pairs
    Currently optimized for Serbian to English translation but could be extended in the future

    Args:
        query: The user's query in the source language
        source_language: Source language code ("sr" for Serbian)
        target_language: Target language code ("en" for English)
        llm: Optional LLM to use. Defaults to configured model if None

    Returns:
        dict keys:
            - original_query: The input query
            - translated_query: The translated query
            - confidence: Translation confidence score (0.0-1.0)
    """

    if source_language == "sr" and target_language == "en":
        result = translate_serbian_to_english(query, llm)
        return {
            "original_query": result.original_query,
            "translated_query": result.translated_query,
            "confidence": result.translation_confidence,
        }

    raise ValueError(
        f"Translation from {source_language} to {target_language} is not yet supported."
    )

