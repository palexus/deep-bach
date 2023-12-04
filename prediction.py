#import openai
import os
from utils import save_piece

# new
from openai import OpenAI


client = OpenAI(
  api_key=os.environ['API_KEY'],  # this is also the default, it can be omitted
)


FINE_TUNED_MODEL = "curie:ft-personal-2022-07-17-00-27-00"
#openai.api_key = os.environ['API_KEY']



PROMPT = """C4 G3 E3 C3
_ _ _ _
_ _ _ B2
_ _ _ _
C4 C4 E3 A2
_ _ _ _
_ _ C3 _
_ _ _ _
D4 C4 G3 G2
_ _ _ _
_ B3 _ _
_ _ _ _
E4 C4 G3 C3
_ _ _ _
_ _ E3 _
_ _ _ _"""


def auto_generate(prompt, steps, seen_lines=64):
    print(len(prompt))
    for _ in range(steps):
        pred_prompt = "\n".join(prompt.split("\n")[-seen_lines:])
        try:
            #print(pred_prompt)
            #output = openai.Completion.create(model=FINE_TUNED_MODEL, prompt=prompt, temperature=0.6)
            #prompt += output["choices"][0]["text"]

            output = client.completions.create(model=FINE_TUNED_MODEL, prompt=prompt, temperature=0.6)
            prompt += output.choices[0].text
            print(output.choices[0].text)
            print(dict(output).get('usage'))
        except Exception as e:
            print("Strange exception occured")
            print(e)
            print(len(prompt))
            break
    return prompt




if __name__=="__main__":
    print("Prediciton starts")
    res = auto_generate(PROMPT, 60)
    print("First round over")
    save_piece(res)
    print("Saved")