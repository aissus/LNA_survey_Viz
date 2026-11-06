"""Every survey the build knows, by id. One line per survey."""

from __future__ import annotations

from survey_sources import LNA, SurveySource

SURVEYS: dict[str, SurveySource] = {s.id: s for s in (LNA,)}
