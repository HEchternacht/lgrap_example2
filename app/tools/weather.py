"""Dummy weather tool — returns simulated weather data for demonstration purposes."""
from __future__ import annotations

import random

from langchain_core.tools import tool

from app.utils.context import tool_config

_CONDITIONS = [
    "sunny",
    "partly cloudy",
    "cloudy",
    "light rain",
    "heavy rain",
    "thunderstorm",
    "snow",
    "windy",
    "foggy",
    "clear",
]


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather conditions for a given city.

    Returns temperature, sky condition, humidity, and wind speed.
    Default unit is Celsius; pass extra_body {"get_weather.temperature": "fahrenheit"}
    in the chat completion request to receive Fahrenheit instead.
    Note: This is a demonstration tool returning simulated data.

    Args:
        city: Name of the city to get weather for (e.g., "London", "Tokyo").
    """
    cfg = tool_config.get().get("get_weather", {})
    use_fahrenheit = cfg.get("temperature", "celsius").lower() == "fahrenheit"

    temp_c = random.randint(-10, 42)
    feels_like_c = temp_c - random.randint(0, 5)
    condition = random.choice(_CONDITIONS)
    humidity = random.randint(20, 98)
    wind_speed = random.randint(0, 90)

    if use_fahrenheit:
        temp = round(temp_c * 9 / 5 + 32, 1)
        feels_like = round(feels_like_c * 9 / 5 + 32, 1)
        unit = "°F"
    else:
        temp = temp_c
        feels_like = feels_like_c
        unit = "°C"

    return (
        f"Weather in {city.title()}:\n"
        f"  Temperature : {temp}{unit} (feels like {feels_like}{unit})\n"
        f"  Condition   : {condition.title()}\n"
        f"  Humidity    : {humidity}%\n"
        f"  Wind Speed  : {wind_speed} km/h\n"
        f"  [simulated data]"
    )
