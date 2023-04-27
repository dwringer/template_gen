import random
import re

from omegaconf import OmegaConf, listconfig, dictconfig
from transformers import pipeline
from transformers.pipelines import text_generation

LOOKUP_TABLE = {'templates': [], 'negatives': []}

generator = None


def prefixer(a : str, b: str):
  "Attach pre/suffixes (indicated w/'@@') and rearrange parens/+/-"
  # print("> " + a + " @@ " + b)  # in case of bugs, uncomment these
  result = None
  if re.match(r"[\)\+\-]+$", b):
    result = a + b + " @@"
  else:
    # Allow a single trailing hyphen but move the rest of ), +, -:
    _aWeighting = re.match(r"\S*?(-?)([\)*\+\-]*)\s*$", a)
    _newA = a
    if _aWeighting is not None:
      _newA = a[0:_aWeighting.start(2)]
      _aWeighting = _aWeighting[2]
    # grab parens from the front of b:
    _bWeighting = re.match(r"\s*(\(*)(\S*)", b)
    _newB = b
    if _bWeighting is not None:
      _newB = _bWeighting[2]
      _bWeighting = _bWeighting[1]
    # now subtract the parens that cancel out:
    _newAWeighting = _aWeighting
    _newBWeighting = ""
    for char in _bWeighting:
      if char == '(':
        _newAWeighting = _aWeighting.replace(")", "", 1)
        if _newAWeighting == _aWeighting:
          _newBWeighting = _newBWeighting + char
        _aWeighting = _newAWeighting
      else:
        _newBWeighting = _newBWeighting + char
    _raw = _newBWeighting + _newA + _newB + _newAWeighting
    # print("> " + _newBWeighting + " || " + _newA + " || " + _newB + " || " + _newAWeighting)
    # then subtract trailing +-'s that cancel
    result = _raw.replace("+-", "", 1)
    while _raw != result:
      _raw = result
      result = _raw.replace("+-", "", 1)
  # print("> " + result)
  return result

def cleanup(string_in : str):
  "Regex to clean formatting typos occurring during generation (TODO: improve)"
  # Condense whitespace
  _str = re.sub(r"\s+", " ", string_in, 0)
  _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) + " "), _str, 0)
  # Remove empty sets of parens
  _str = re.sub(r"\(\)[\+\-]*", "", _str)
  # Condense whitespace again
  _str = re.sub(r"\s+", " ", _str, 0)
  _str = re.sub(r"\s([,;])\s", lambda match: (match.group(1) +  " "), _str, 0)
  _str = re.sub(r",+\s", ", ", _str, 0)
  # Now we attempt to combine prefixes and suffixes.
  _collapsed = re.sub(r"(\S+?)\s*@@\s*(\S+)", lambda match: prefixer(match.group(1), match.group(2)), _str, 0)
  while _collapsed != _str:
    _str = _collapsed
    _collapsed = re.sub(r"(\S+?)\s*@@\s*(\S+)", lambda match: prefixer(match.group(1), match.group(2)), _str, 0)
  # Remove instances of "a apple" errors by subbing "an", w/ or w/o parens in the way.
  string_out = re.sub(r"(\sa\s)([\(]*)([aeiouAEIOU])", lambda match: (" an " + match.group(2) + match.group(3)), _str, 0)
  return string_out

def addNegatives(prompts : list):
  "Adds randomly chosen negatives to a list of prompts"
  return [(p + " " + random.choice(LOOKUP_TABLE['negatives'])) for p in prompts]

def parsedTemplateLines(lines : list):
  "Parse the rows of a template key, creating copies etc."
  results = []
  for line in lines:
    if isinstance(line, str):
      if (0 < len(line)) and (line[0] == "\\"):  # We use backslash to escape parsing
        results.append(line[1:])
      else:
        # - N * ..., where N is a number, adds N copies of prompt "..."
        _match = re.match(r"\s*(\d+)\s*\*\s*(.*)", line)
        if _match is not None:
          for i in range(int(_match[1])):
            results.append(_match[2])
        # - ?N, where N is a number, is treated as an instruction to add N "empty" choices:
        elif re.match(r"\?[\d]+$", line):
          results.extend(["" for number_of_times in range(int(line[1:]))])
        else:
          results.append(line)
    elif isinstance(line, listconfig.ListConfig) and (len(line) == 3):
      for i in range(int(line[0])):
        results.append(line[1:])
    else:
      results.append(line)
  return results

def loadTemplate(filename : str):
  "This loads a template file following the pattern of example_template.yaml"
  # This can be used to load invokeai-batch YAML templates as addl. words/template forms
  _conf = OmegaConf.load(filename)
  _prompt = _conf.get("prompt")
  if isinstance(_prompt, dictconfig.DictConfig):
    for k, v in _prompt.items():
      if (k == "template"):
        if 'templates' not in LOOKUP_TABLE:
          LOOKUP_TABLE['templates'] = []
        LOOKUP_TABLE['templates'].append(v)
      elif (k == "negative"):
        if 'negatives' not in LOOKUP_TABLE:
          LOOKUP_TABLE['negatives'] = []
        LOOKUP_TABLE['negatives'].append(v)
      else:
        if (k not in LOOKUP_TABLE):
          LOOKUP_TABLE[k] = []
        LOOKUP_TABLE[k].extend(parsedTemplateLines(v))

def templateExpand(s :          str,
                   lookups :    dict = LOOKUP_TABLE,
                   reflection : str  = ""):
  "Used internally to replace words with their template lookups. Single pass."
  _split = re.split(r'({\w+})', s)
  result = ""
  for word in _split:
    if re.fullmatch(r'({\w+})', word):
      _lookup = random.choice(lookups[word[1:-1]])
      _resultAppendix, _reflectionPrependix = None, None
      if isinstance(_lookup, (list, listconfig.ListConfig)):
        _resultAppendix = _lookup[0]
        _reflectionPrependix = _lookup[1]
#        result = result + _lookup[0]
#        reflection = " " + _lookup[1] + reflection
      else:
        _resultAppendix = _lookup
#        result = result + _lookup
      result = result + _resultAppendix
      if _reflectionPrependix is not None:
        reflection = " " + _reflectionPrependix + reflection
    else:
      result = result + word
  return result, reflection

def makePrompts(n :                        int,
                lookups :                  dict  = LOOKUP_TABLE,
                template_strings :         list  = None,
                remove_negatives :         bool  = False,
                base_negatives :           list  = None,
                strip_parens_probability : float = 0.0):
  "This function returns a list of prompts generated from loaded templates."
  results = []
  if template_strings is None:
    template_strings = lookups['templates']
  if base_negatives is None:
    base_negatives = lookups['negatives']
  for i in range(n):
    _str, _reflection = templateExpand(random.choice(template_strings), reflection="")
    _next, _reflection = templateExpand(_str, lookups=lookups, reflection=_reflection)
    while ((_next != _str) or (re.search(r'{\w+}', _str))):
      _str = _next
      _next, _reflection = templateExpand(_str, lookups=lookups, reflection=_reflection)
    # _reflection is a parallel prompt built in reverse during expansion
    while _reflection != "":
      _appendix = _reflection
      _next, _reflection = templateExpand(_appendix, lookups=lookups, reflection="")
      while ((_next != _appendix) or (re.search(r'{\w+}', _appendix))):
        _appendix = _next
        _next, _reflection = templateExpand(_appendix, lookups=lookups, reflection=_reflection)
      _str = _str + _appendix

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
    results.append(_str.strip())
  return results

def loadPipeline(model :     str,
                 tokenizer : str = None,
                 task :      str = 'text-generation',
                 device :    int = -1):  # device=0 will use CUDA
  "Load a transformers text generation pipeline from the given folder/repo name"
  global generator  # We keep one cached globally
  if tokenizer is None:
    tokenizer = model
  generator = pipeline(model=model, tokenizer=tokenizer, task=task, device=device)
  return generator

def makePromptsP(prompt :   str   = "",
                 temp :     float = 1.4,
                 k :        int   = 40,
                 p :        float = 0.9,
                 n :        int   = 20,
                 mnt :      int   = 150,
                 pipeline : text_generation.TextGenerationPipeline = None ):
  "Use cached or provided transformers text generation pipeline to make prompts"
  global generator
  if pipeline is None:
    pipeline = generator
  _outputs = pipeline(prompt,
                      max_new_tokens=mnt,
                      temperature=temp,
                      do_sample=True,
                      top_p=p,
                      top_k=k,
                      num_return_sequences=n)
  items = set([cleanup(re.sub(r"\n", " ", output['generated_text'], 0)) for output in _outputs])
  return items

def printTemplate(filename :      str,
                  prompts :       list  = None,
                  sampler :       str   = "k_dpmpp_2",
                  cfg :           str   = "7",
                  steps :         int   = 39,
                  width :         int   = 512,
                  height :        int   = 512,
                  perlin :        float = 0,
                  threshold :     float = 0,
                  seed_attempts : int   = 1,
                  models :        list  = ["526mixV145_v145"],
                  hires_fix :     float = None,
                  args :          str   = None,
                  seed :          int   = None):
  "This function prints a template file in the same format as invokeai-batch outputs, run with `invoke --from_file filename`"
  if args is None:
    args = "-A" + sampler + " -C" + str(cfg) + " -s" + str(steps) + "  --perlin=" + str(perlin) + " --threshold=" + str(threshold) + " -W" + str(width) + " -H" + str(height)
    if hires_fix is not None:
      args = args + " --hires_fix -f" + str(hires_fix)
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
