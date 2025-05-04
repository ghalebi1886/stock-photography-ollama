from crewai import Agent
# Note: Tool instances will be assigned when the agent is instantiated in main.py
# from src.tools.metadata_extractor_tool import MetadataExtractorTool

class MetadataSpecialistAgents:
    """
    Contains the definition for the Senior Technical Metadata Specialist agent.
    """
    def senior_technical_metadata_specialist(self) -> Agent:
        """
        Defines the Senior Technical Metadata Specialist agent.
        This agent extracts key EXIF data from images.
        """
        return Agent(
            role="Senior Technical Metadata Specialist",
            goal="To meticulously extract and structure essential EXIF technical metadata "
                 "from image files, ensuring data accuracy and completeness for downstream processing.",
            backstory=(
                "With years of experience handling diverse digital image formats and camera systems, "
                "you are the go-to expert for reliably extracting technical footprints (EXIF data). "
                "You understand that accurate metadata is the bedrock upon which all further image "
                "analysis is built. Precision and consistency are your hallmarks."
            ),
            # tools=[MetadataExtractorTool.extract_metadata], # Assign tool instance in main workflow
            allow_delegation=False,
            verbose=True,
            memory=False # This agent likely doesn't need memory between tasks on different images
        )

# Example of how to potentially instantiate (for reference, not used directly here)
# if __name__ == '__main__':
#     from src.tools.metadata_extractor_tool import MetadataExtractorTool
#     agent_defs = MetadataSpecialistAgents()
#     specialist = agent_defs.senior_technical_metadata_specialist()
#     # In a real scenario, you'd likely pass the tool instance here if needed immediately
#     # specialist.tools = [MetadataExtractorTool()] # Example assignment
#     print("Agent Definition:")
#     print(f"Role: {specialist.role}")
#     print(f"Goal: {specialist.goal}")
#     # print(f"Tools: {specialist.tools}")
