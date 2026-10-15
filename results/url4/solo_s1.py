import screamingface as sf

candidate = sf.Model(
    'gemini-cli/gemini-3.1-flash-lite',
    name='solo_s1',
    params={'temperature': 0.7},
)

report = sf.evaluate(
    candidate,
    benchmark='ifeval',
    limit=50,
)