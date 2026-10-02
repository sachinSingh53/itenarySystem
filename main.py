"""Generate a schema-validated travel itinerary with Google Gemini."""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, model_validator


MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class DailyPlan(BaseModel):
    """Activities planned for one day."""

    model_config = ConfigDict(extra="forbid")

    day: int
    activities: list[str]


class TravelItinerary(BaseModel):
    """The exact JSON contract requested for a generated itinerary."""

    model_config = ConfigDict(extra="forbid")

    destination: str
    trip_duration_days: int
    budget_category: str
    top_attractions: list[str]
    daily_plan: list[DailyPlan]

    @model_validator(mode="after")
    def validate_daily_plan(self) -> TravelItinerary:
        """Ensure the day count and numbered plans agree."""
        if self.trip_duration_days < 1:
            raise ValueError("trip_duration_days must be positive")

        expected_days = list(range(1, self.trip_duration_days + 1))
        actual_days = [plan.day for plan in self.daily_plan]
        if actual_days != expected_days:
            raise ValueError(
                "daily_plan must contain one ordered entry for every trip day"
            )

        if not self.top_attractions:
            raise ValueError("top_attractions must not be empty")
        if any(not plan.activities for plan in self.daily_plan):
            raise ValueError("each day must contain at least one activity")

        return self


def prompt_for_city(label: str) -> str:
    """Prompt until the user enters a non-empty city name."""
    while True:
        city = input(f"Enter {label} city: ").strip()
        if city:
            return city
        print("City name cannot be empty.")


def get_city_pair() -> tuple[str, str]:
    """Read distinct source and destination cities from the terminal."""
    source = prompt_for_city("source")

    while True:
        destination = prompt_for_city("destination")
        if destination.casefold() != source.casefold():
            return source, destination
        print("Destination city must be different from the source city.")


def generate_itinerary(
    client: genai.Client,
    source: str,
    destination: str,
) -> TravelItinerary:
    """Ask the model for an itinerary parsed into ``TravelItinerary``."""
    system_instruction = (
        "You are a practical travel planner. Use only your own general "
        "knowledge: do not use search or claim live prices, opening hours, "
        "availability, weather, or other real-time facts. Choose a sensible "
        "trip duration from 3 to 7 days and a concise budget category. The "
        "daily_plan must contain exactly one ordered item for each day, "
        "numbered from 1 through trip_duration_days."
    )
    prompt = (
        f"Create a trip from {source} to {destination}. "
        f'The destination field must be exactly "{destination}".'
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_json_schema=TravelItinerary.model_json_schema(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini did not return an itinerary.")

    return TravelItinerary.model_validate_json(response.text)


def main() -> None:
    """Read two cities and generate their itinerary."""
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before running this program."
        )

    source, destination = get_city_pair()
    itinerary = generate_itinerary(genai.Client(), source, destination)
    print(itinerary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
