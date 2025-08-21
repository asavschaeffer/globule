"""Debug Mode for the Glass Engine."""

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode

class DebugGlassEngine(AbstractGlassEngine):
    def get_mode(self) -> GlassEngineMode:
        return GlassEngineMode.DEBUG

    async def execute_tutorial_flow(self) -> None:
        self.console.print("[bold yellow]Debug Mode is not yet implemented.[/bold yellow]")
