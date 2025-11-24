# Darwin-Gödel Machine: Self-Improving Molecular Evolution

## What This Does

Evolves drug-like molecules using AI and evolution. The system mutates molecules, keeps the good ones, and even improves its own mutation strategy over time.

## Example Results

Here's what the molecules look like when validated:

![BOILED-Egg Plot](images/egg.png)
*Absorption prediction - white area means good oral absorption*

![Molecular Properties](images/molecule.png)
*Drug-likeness analysis - green means good, red means bad*

![Target Classes](images/target_classes.png)
*What proteins the molecule might bind to*

![Target Predictions](images/similarity.png)
*Specific protein targets with confidence scores*

## How It Works

1. Start with random molecules
2. Mutate them (randomly + with AI every 10 generations)
3. Keep the best ones based on drug-likeness score
4. When stuck, explore weird/novel molecules instead
5. Every 50 generations, the AI updates its own mutation strategy

That's it. It's basically evolution with AI assistance and self-improvement.

## Installation

```bash
export GEMINI_API_KEY="your-api-key"
pip install rdkit-pypi numpy pandas matplotlib google-genai
python3 main.py
```

## Output

You get:
- Best molecules found (`molecules/`)
- Fitness plots (`plots/`)
- Evolution logs (`snapshots/`)

## Validate Your Molecules

1. Go to https://www.swissadme.ch/
2. Paste your SMILES
3. Check if it looks drug-like

For target prediction:
1. Go to https://www.swisstargetprediction.ch/
2. Paste SMILES
3. See what proteins it might hit

## Key Concepts

- **QED Score**: 0-1, higher = more drug-like
- **Novelty**: How different from previously seen molecules
- **Plateau**: When evolution gets stuck, switches to exploring novelty
- **Self-improvement**: The AI prompt evolves based on what works

## Parameters

Edit in the code:
- `POPULATION_SIZE = 10` - molecules per generation
- `LLM_MUTATION_INTERVAL = 10` - when to use AI mutations
- `PROMPT_REFINEMENT_INTERVAL = 50` - when to update the prompt
- `PLATEAU_THRESHOLD = 30` - generations without improvement = stuck

## License

MIT