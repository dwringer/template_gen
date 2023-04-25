import random
import re

from omegaconf import OmegaConf, dictconfig


LOOKUP_TABLE = {}
TEMPLATES = []
NEGATIVES = []

generator = None


def loadPipeline(model, tokenizer, task='text-generation', device=-1):  # device=0 will use CUDA
  from transformers import pipeline

  return pipeline(model=model, tokenizer=tokenizer, task=task, device=device)

def cleanup(str):
    # Condense whitespace
    _str = re.sub(r"\s+", " ", str, 0)
    _str = re.sub(r"\s,\s", ", ", _str, 0)
    # Remove empty sets of parens
    _str = re.sub(r"\(\)[\+\-]*", "", _str)
    # Condense whitespace again
    _str = re.sub(r"\s+", " ", _str, 0)
    _str = re.sub(r"\s,\s", ", ", _str, 0)
    _str = re.sub(r",+\s", ", ", _str, 0)
    # Remove instances of "a apple" errors by subbing "an", w/ or w/o parens in the way.
    _str = re.sub(r"(\sa\s)([\(]*)([aeiouAEIOU])", lambda match: (" an " + match.group(2) + match.group(3)), _str)
    return _str

def addNegatives(prompts):
  return [(p + " " + random.choice(NEGATIVES)) for p in prompts]
  
def makePromptsP(prompt: str = "",
                 top_p: float = 0.9,
                 top_k: int = 40,
                 n: int = 20,
                 temp: float = 1.4,
                 max_new_tokens: int = 150):
  outputs = generator(prompt,
                      max_new_tokens=max_new_tokens,
                      temperature=temp,
                      do_sample=True,
                      top_p=top_p,
                      top_k=top_k,
                      num_return_sequences=n)
  items = set([cleanup(re.sub(r"\n", " ", output['generated_text'], 0)) for output in outputs])
  return items

def loadTemplate(filename):
  "This loads a template file following the pattern of example_template.yaml"
  # This can be used to load invokeai-batch YAML templates as addl. words/template forms
  _conf = OmegaConf.load(filename)
  _prompt = _conf.get("prompt")
  if isinstance(_prompt, dictconfig.DictConfig):
    for k, v in _prompt.items():
      if (k == "template"):
        TEMPLATES.append(v)
      elif (k == "templates"):
        TEMPLATES.extend(v)
      elif (k == "negative"):
        NEGATIVES.append(v)
      elif (k == "negatives"):
        NEGATIVES.extend(v)
      else:
        if (k not in LOOKUP_TABLE):
          LOOKUP_TABLE[k] = []
        # - ?N, where N is a number, is treated as an instruction to add N "empty" choices:
        if isinstance(v[0], str) and (re.match(r"\?[\d]+$", v[0]) is not None):
          LOOKUP_TABLE[k].extend(["" for number_of_times in range(int(v[0][1:]))])
          LOOKUP_TABLE[k].extend([subvalue for subvalue in v[1:]])
        else:
          LOOKUP_TABLE[k].extend(v)

def makePrompts(n,
                lookups=LOOKUP_TABLE,
                template_string=None,
                remove_negatives=False,
                base_negatives=NEGATIVES,
                strip_parens_probability=0.0):
  "This function returns a list of prompts generated from whatever templates have been loaded."
  results = []
  for i in range(n):
    # The "bookends" key from template yaml's contains [before, after] pairs to surround prompts:
    _camera = random.choice(lookups["bookends"])
    _str = _camera[0] + " "
    # We grab a random template then use the same logic as dynamic_prompts.py
    template_string = random.choice(TEMPLATES)
    _templateSplit = re.split(r'({\w+})', template_string)
    for word in _templateSplit:
      if re.fullmatch(r'({\w+})', word):
        _str = _str + random.choice(lookups[word[1:-1]])
      else:
        _str = _str + word
    _str = _str + " " + _camera[1]

    # Regex stuff, hastily implemented:
    if random.random() < strip_parens_probability:
      # Strip off parentheses and pluses, then trailing minuses w/o and w/ commas afterward
      _str = re.sub(r"[\(\)+]", "", _str, 0)
      _str = re.sub(r"\-+\s", " ", _str, 0)
      _str = re.sub(r"\-+,", ",", _str, 0)
    if remove_negatives:
      # strip out anything between []
      _str = re.sub(r"\[[^\]\[]*\]", "", _str, 0)
    _str = cleanup(_str)

    if not (remove_negatives or (not base_negatives)):
      _str = _str + " " + random.choice(base_negatives)
    results.append(_str)
  return results

def printTemplate(filename,
                  prompts=None,
                  sampler="k_dpmpp_2",
                  cfg="6.5",
                  steps=42,
                  width=512,
                  height=512,
                  perlin=0,
                  threshold=0,
                  models=["526mixV145_v145"],
                  args=None,
                  seed=None):
  "This function prints a template file in the same format as invokeai-batch outputs, run with `invoke --from_file filename`"
  if args is None:
    args = "-A" + sampler + " -C" + str(cfg) + " -s" + str(steps) + "  --perlin=" + str(perlin) + " --threshold=" + str(threshold) + " -W" + str(width) + " -H" + str(height)
  if prompts is None:
    prompts = makePrompts(300)
  promptLines = [(p + " " + "-S" + (str(seed if seed is not None else random.randint(0, 1000000000))) + " " + args) for p in prompts]
  with open(filename, 'w', encoding='utf8') as outf:
    for model in models:
      outf.writelines(["!switch " + model + '\n'])
      outf.writelines([line + '\n' for line in promptLines])

    
    
