from crewai import Agent
# Note: Tool instances will be assigned when the agent is instantiated in main.py
# from src.tools.technical_analyzer_tool import BasicTechnicalAnalyzerTool

class QualityInspectorAgents:
    """
    Contains the definition for the Chief Image Quality Inspector agent.
    """
    def chief_image_quality_inspector(self) -> Agent:
        """
        Defines the Chief Image Quality Inspector agent.
        This agent assesses technical quality (focus, exposure).
        """
        return Agent(
            role="Chief Image Quality Inspector",
            goal="To critically evaluate the fundamental technical quality (focus and exposure) "
                 "of each image, making a decisive judgment on its suitability for further "
                 "consideration based on defined objective standards.",
            backstory=(
                "You are a highly experienced image quality control professional with a sharp eye "
                "for technical flaws that render an image unusable for professional stock purposes. "
                "You don't deal with subjective aesthetics, only the hard facts of focus and exposure. "
                "Your judgment forms the crucial first gate in the quality pipeline."
            ),
            # tools=[BasicTechnicalAnalyzerTool.perform_analysis], # Assign tool instance in main workflow
            allow_delegation=False,
            verbose=True,
            memory=False # No memory needed between images
        )

# Example of how to potentially instantiate (for reference, not used directly here)
# if __name__ == '__main__':
#     from src.tools.technical_analyzer_tool import BasicTechnicalAnalyzerTool
#     agent_defs = QualityInspectorAgents()
#     inspector = agent_defs.chief_image_quality_inspector()
#     # inspector.tools = [BasicTechnicalAnalyzerTool()] # Example assignment
#     print("Agent Definition:")
#     print(f"Role: {inspector.role}")
#     print(f"Goal: {inspector.goal}")
#     # print(f"Tools: {inspector.tools}")
