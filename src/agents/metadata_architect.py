from crewai import Agent
# Note: Tool instances will be assigned when the agent is instantiated in main.py
# from src.tools.image_describer_tool import ImageDescriberKeywordTool

class MetadataArchitectAgents:
    """
    Contains the definition for the Principal Stock Metadata Architect agent.
    """
    # Make llm parameter optional as it will be assigned after instantiation
    def principal_stock_metadata_architect(self, llm=None) -> Agent:
        """
        Defines the Principal Stock Metadata Architect agent.
        This agent generates stock-optimized titles/descriptions and keywords.

        Args:
            llm: The language model instance or config dictionary (optional during init).
        """
        return Agent(
            role="Principal Stock Metadata Architect",
            goal="To maximize the commercial potential and discoverability of the final selected images "
                 "by crafting expert-level titles/descriptions and generating comprehensive, relevant "
                 "keyword sets optimized for leading stock photo agencies.",
            backstory=(
                "You are a veteran strategist in stock photo metadata optimization, deeply understanding "
                "what drives visibility and sales on platforms like Shutterstock, Adobe Stock, and Getty. "
                "You transform visual content into highly searchable assets by generating concise, "
                "compelling titles and a rich set of commercially valuable keywords, leveraging advanced "
                "AI assistance but applying your expert judgment to ensure relevance and quality."
            ),
            # tools=[ImageDescriberKeywordTool.generate_text_from_image], # Assign tool instance in main workflow
            allow_delegation=False,
            verbose=True,
            memory=False, # No memory needed between images
            llm=llm # Pass the provided llm to the Agent constructor
        )

# Example of how to potentially instantiate (for reference, not used directly here)
# if __name__ == '__main__':
#     from src.tools.image_describer_tool import ImageDescriberKeywordTool
#     agent_defs = MetadataArchitectAgents()
#     architect = agent_defs.principal_stock_metadata_architect()
#     # architect.tools = [ImageDescriberKeywordTool()] # Example assignment
#     print("Agent Definition:")
#     print(f"Role: {architect.role}")
#     print(f"Goal: {architect.goal}")
#     # print(f"Tools: {architect.tools}")
