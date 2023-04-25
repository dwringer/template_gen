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

def cleanup(string_in):
    # Condense whitespace
    _str = re.sub(r"\s+", " ", string_in, 0)
    _str = re.sub(r"\s,\s", ", ", _str, 0)
    # Remove empty sets of parens
    _str = re.sub(r"\(\)[\+\-]*", "", _str)
    # Condense whitespace again
    _str = re.sub(r"\s+", " ", _str, 0)
    _str = re.sub(r"\s,\s", ", ", _str, 0)
    _str = re.sub(r",+\s", ", ", _str, 0)
    # Remove instances of "a apple" errors by subbing "an", w/ or w/o parens in the way.
    string_out = re.sub(r"(\sa\s)([\(]*)([aeiouAEIOU])", lambda match: (" an " + match.group(2) + match.group(3)), _str)
    return string_out

def addNegatives(prompts):
  return [(p + " " + random.choice(NEGATIVES)) for p in prompts]
  
def makePromptsP(prompt: str = "",
                 p: float = 0.9,
                 k: int = 40,
                 n: int = 20,
                 temp: float = 1.4,
                 max_new_tokens: int = 150):
  _outputs = generator(prompt,
                       max_new_tokens=max_new_tokens,
                       temperature=temp,
                       do_sample=True,
                       top_p=p,
                       top_k=k,
                       num_return_sequences=n)
  items = set([cleanup(re.sub(r"\n", " ", output['generated_text'], 0)) for output in _outputs])
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
        if isinstance(v[0], str) and (re.match(r"\?[\d]+$", v[0]) is not None):
          NEGATIVES.extend(["" for number_of_times in range(int(v[0][1:]))])
          NEGATIVES.extend([subvalue for subvalue in v[1:]])
        else:
          NEGATIVES.extend(v)
      else:
        if (k not in LOOKUP_TABLE):
          LOOKUP_TABLE[k] = []
        # - ?N, where N is a number, in slot 1, is treated as an instruction to add N "empty" choices:
        if isinstance(v[0], str) and (re.match(r"\?[\d]+$", v[0]) is not None):
          LOOKUP_TABLE[k].extend(["" for number_of_times in range(int(v[0][1:]))])
          LOOKUP_TABLE[k].extend([subvalue for subvalue in v[1:]])
        else:
          LOOKUP_TABLE[k].extend(v)

def templateExpand(s, lookups=LOOKUP_TABLE):
  _split = re.split(r'({\w+})', s)
  result = ""
  for word in _split:
    if re.fullmatch(r'({\w+})', word):
      result = result + random.choice(lookups[word[1:-1]])
    else:
      result = result+ word
  return result

def makePrompts(n,
                lookups=LOOKUP_TABLE,
                template_strings=None,
                remove_negatives=False,
                base_negatives=NEGATIVES,
                strip_parens_probability=0.0):
  "This function returns a list of prompts generated from whatever templates have been loaded."
  results = []
  if template_strings is None:
    template_strings = TEMPLATES
  for i in range(n):
    # The "bookends" key from template yaml's contains [before, after] pairs to surround prompts:
    _camera = random.choice(lookups["bookends"])
    _str = templateExpand(_camera[0] + " " + random.choice(template_strings) + " " + _camera[1], lookups=lookups)
    _next = templateExpand(_str, lookups=lookups)
    while (_next != _str):
      _str = _next
      _next = templateExpand(_str, lookups=lookups)

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
                  seed_attempts=1,
                  models=["526mixV145_v145"],
                  args=None,
                  seed=None):
  "This function prints a template file in the same format as invokeai-batch outputs, run with `invoke --from_file filename`"
  if args is None:
    args = "-A" + sampler + " -C" + str(cfg) + " -s" + str(steps) + "  --perlin=" + str(perlin) + " --threshold=" + str(threshold) + " -W" + str(width) + " -H" + str(height)
  if prompts is None:
    prompts = makePrompts(300)
  promptLines = []
  for p in prompts:
    if seed is not None:
      promptLines.append(p + " " + "-S" + str(seed) + " " + args)
    else:
      promptLines.extend([(p + " -S" + str(random.randint(0, 1000000000)) + " " + args) for i in range(seed_attempts)])
  with open(filename, 'w', encoding='utf8') as outf:
    for model in models:
      outf.writelines(["!switch " + model + '\n'])
      outf.writelines([line + '\n' for line in promptLines])
