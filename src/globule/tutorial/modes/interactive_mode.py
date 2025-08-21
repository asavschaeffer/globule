"""Interactive Mode for the Glass Engine."""

from ..glass_engine_core import AbstractGlassEngine, GlassEngineMode

class InteractiveGlassEngine(AbstractGlassEngine):
    def get_mode(self) -> GlassEngineMode:
        return GlassEngineMode.INTERACTIVE

    async def execute_tutorial_flow(self) -> None:
        self.console.print("[bold yellow]Interactive Mode is not yet implemented.[/bold yellow]")
