This is a prompt generator that uses templates to randomly generate prompt permutations and, optionally, make them into invoke-readable input files.

Create a folder inside your invokeai/ directory called "gen", or whatever you prefer, and place these files in it.
- template_gen.py
- example_template.yaml

Now load the invoke developer console:

    C:\[path-to]\invokeai>invoke.bat
 
    Do you want to generate images using the
    1. command-line interface
    2. browser-based UI
    3. run textual inversion training
    4. merge models (diffusers type only)
    5. download and install models
    6. change InvokeAI startup options
    7. re-run the configure script to fix a broken install
    8. open the developer console <-- choose 8 to enter the virtual environment
    9. update InvokeAI
    10. command-line help
    Q - quit
    Please enter 1-10, Q: [2] 8

    (.venv) C:\[path-to]\invokeai>cd gen
    (.venv) C:\[path-to]\invokeai\gen>python -i template_gen.py

    >>> loadTemplate("example_template.yaml")

    >>> for p in makePrompts(5): print(p)
    ...
    professional photo of a (happy)+ (man)++ (transforming+++ into an eyeless++ soulless+++ mess of wires and circuits and vacuum tubes and black plastic)-, 50mm f/5.6 on fujicolor pro 800z [sketch, drawing, cartoon, pixar, zombie--]
    professional photo of a (man)++ (transforming+++ into an eyeless++ soulless+++ mess of wires and circuits and vacuum tubes and black plastic)-, 50mm f/5.6 on fujicolor pro 800z
    professional photo of a (Rami Malek)++ (transforming+++ into an eyeless++ soulless+++ mess of wires and circuits and vacuum tubes and black plastic)-, 50mm f/5.6 on fujicolor pro 800z [sketch, drawing, cartoon, pixar, zombie--]
    oil painting using impasto of a (Rami Malek)++ (in a futuristic drinks bar of a space station)-, brush strokes, palette knife technique [sketch, drawing, cartoon, pixar, zombie--]
    oil painting using impasto of a (man)++ (in a futuristic drinks bar of a space station)-, brush strokes, palette knife technique [sketch, drawing, cartoon, pixar, zombie--]

    >>> printTemplate("example_output.txt", makePrompts(10), models=["526mixV14_v14", "verisimilitude"])

The above call to printTemplate assumes you have the models 526mixV14_v14 and verisimilitude installed. Change as needed! Now, from a new terminal, go to the path containing the invoke executable (invokeai\\.venv\\Scripts on Windows) and run invoke specifying the path to the template thus created.

    C:\[path-to]\invokeai\.venv\Scripts>invoke --from_file ../../gen/example_output.txt
