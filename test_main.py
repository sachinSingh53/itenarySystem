"""Focused tests for itinerary selection and structured parsing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from main import TravelItinerary, generate_itinerary, get_city_pair


class ItineraryTests(unittest.TestCase):
    @patch("builtins.input", side_effect=["Jaipur", "Kyoto"])
    def test_city_pair_is_read_from_terminal(self, _mock_input: Mock) -> None:
        source, destination = get_city_pair()

        self.assertEqual(source, "Jaipur")
        self.assertEqual(destination, "Kyoto")

    @patch(
        "builtins.input",
        side_effect=["", "Jaipur", "jaipur", "Kyoto"],
    )
    def test_city_pair_rejects_blank_and_matching_names(
        self, _mock_input: Mock
    ) -> None:
        source, destination = get_city_pair()

        self.assertEqual(source, "Jaipur")
        self.assertEqual(destination, "Kyoto")

    def test_generation_uses_pydantic_schema(self) -> None:
        parsed = TravelItinerary.model_validate(
            {
                "destination": "Kyoto, Japan",
                "trip_duration_days": 2,
                "budget_category": "mid-range",
                "top_attractions": ["Fushimi Inari Taisha"],
                "daily_plan": [
                    {"day": 1, "activities": ["Explore Gion"]},
                    {"day": 2, "activities": ["Visit Arashiyama"]},
                ],
            }
        )
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text=parsed.model_dump_json()
        )

        result = generate_itinerary(
            client,
            source="Jaipur, India",
            destination="Kyoto, Japan",
        )

        self.assertEqual(result, parsed)
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNone(config.response_schema)
        self.assertEqual(
            config.response_json_schema, TravelItinerary.model_json_schema()
        )
        self.assertFalse(config.response_json_schema["additionalProperties"])
        self.assertFalse(
            config.response_json_schema["$defs"]["DailyPlan"][
                "additionalProperties"
            ]
        )
        self.assertNotIn("additional_properties", str(config.response_json_schema))
        self.assertTrue(config.automatic_function_calling.disable)

    def test_schema_rejects_missing_days(self) -> None:
        with self.assertRaisesRegex(ValueError, "one ordered entry"):
            TravelItinerary.model_validate(
                {
                    "destination": "Lisbon, Portugal",
                    "trip_duration_days": 2,
                    "budget_category": "budget",
                    "top_attractions": ["Belém Tower"],
                    "daily_plan": [
                        {"day": 1, "activities": ["Walk through Alfama"]}
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
