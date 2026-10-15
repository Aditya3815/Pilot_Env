import screamingface as sf

candidate = sf.Model(
    'gemini-cli/gemini-3.1-flash-lite',
    name='solo_default',
)

report = sf.evaluate(
    candidate,
    benchmark='ifeval',
    limit=50,
)