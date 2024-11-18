from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


information = """
Jonathan Michael Batiste (born November 11, 1986)[2] is an American singer, songwriter, multi-instrumentalist, bandleader, composer, and television personality.[3] He has recorded and performed with artists including Stevie Wonder, Prince, Willie Nelson, Lenny Kravitz, ASAP Rocky, Ed Sheeran, Lana Del Rey, Roy Hargrove, Juvenile, and Mavis Staples. Batiste appeared nightly with his band, Stay Human,[4] as bandleader and musical director on The Late Show with Stephen Colbert from 2015 to 2022.[5][6] Batiste also serves as the music director of The Atlantic and the Creative Director of the National Jazz Museum in Harlem. In 2020, he co-composed the score for the Pixar animated film Soul, for which he received an Academy Award, a Golden Globe Award, a Grammy Award and a BAFTA Film Award (all shared with Trent Reznor and Atticus Ross).[7] Batiste has garnered five Grammy Awards from 20 nominations, including an Album of the Year win for We Are (2021).[8] In 2023, Batiste was featured in the documentary film, American Symphony, which records the process of Batiste composing his first symphony.[9] In 2024, Batiste featured in the ensemble comedy-drama film Saturday Night, directed by Jason Reitman, playing the role of musician Billy Preston, as well as composing the film's score.[10]
"""


if __name__ == "__main__":
    load_dotenv()
    summary_template = """
        given the information {information} about a person, I want you to output:
        1. their name
        2. a short summary
        3. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm

    print(information)

    # res = chain.invoke(input={"information": information})
    res = chain.invoke(input={"information": information})
    print(res)
