from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import diskcache
import os
import dotenv

dotenv.load_dotenv()

cache = diskcache.FanoutCache(f"{os.getenv("CACHE_FOLDER")}/crime_description_cache")

class BehaviorDescriptionAgent:
    def __init__(self, openai_api_key):
        self.system_prompt = """
        You are an AI assistant that generates a single, concise, and clear behavior description that integrates two Android APIs.
        Follow these structured rules:

        1. Start the description with a verb in simple present tense (e.g., Get, Read, Store, Send, Save).
        2. Use simple present tense throughout the description.
        3. Keep it concise—prefer a single verb where possible.
        4. Simplify UI-related terminology for general understanding.
        5. Maintain a consistent pattern: 'Action + Object + Purpose (if necessary)'.
        6. Integrate both APIs into **one** meaningful behavior description instead of generating separate descriptions.

        **Examples:**
        - API 1: Landroid/content/Context;.getPackageName ()Ljava/lang/String;
          API 2: Landroid/app/AlertDialog$Builder;.setAdapter (Landroid/widget/ListAdapter; Landroid/content/DialogInterface$OnClickListener;)Landroid/app/AlertDialog$Builder;
          **Generated Behavior:** Get the package name and set it as an adapter in an AlertDialog.

        - API 1: Landroid/app/Dialog;.findViewById (I)Landroid/view/View;
          API 2: Landroid/content/SharedPreferences$Editor;.putString (Ljava/lang/String; Ljava/lang/String;)Landroid/content/SharedPreferences$Editor;
          **Generated Behavior:** Store a dialog view’s value in SharedPreferences.

        Now, for each of the following messages, I will provide a pair of APIs, please refer to the above examples, and generate a single behavior description.
        """
        self.user_prompt_template = (
            "- API 1: {api1[class]}.{api1[method]} {api1[descriptor]}\n"
            "- API 2: {api2[class]}.{api2[method]} {api2[descriptor]}"
        )
        self.openai_api_key = openai_api_key
        self.chat_model = ChatOpenAI(model_name="gpt-4.1-nano", openai_api_key=self.openai_api_key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.chat_model = None

    @cache.memoize(ignore={"self", 0})
    def get_description(self, api_pair) -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.user_prompt_template.format(api1=api_pair[0], api2=api_pair[1]))
        ]
        return self.chat_model(messages).content.strip()

# Example Usage
if __name__ == "__main__":
    openai_api_key = "your_openai_api_key_here"  # Replace with your OpenAI API key

    api1 = {
        "class": "Landroid/app/Dialog;",
        "method": "setCancelable",
        "descriptor": "(Z)V"
    }
    api2 = {
        "class": "Landroid/content/SharedPreferences$Editor;",
        "method": "putString",
        "descriptor": "(Ljava/lang/String; Ljava/lang/String;)Landroid/content/SharedPreferences$Editor;"
    }

    with BehaviorDescriptionAgent(openai_api_key) as agent:
        description = agent.get_description((api1, api2))
        print("Generated Behavior Description:", description)
