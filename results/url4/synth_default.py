import screamingface as sf

candidate = sf.Fusion(
    [
        sf.Model(
            'gemini-cli/gemini-3.1-flash-lite',
            params={'temperature': 0.7},
        ),
        sf.Model(
            'gemini-cli/gemini-3.1-flash-lite',
            params={'temperature': 0.7},
        ),
        sf.Model(
            'gemini-cli/gemini-3.1-flash-lite',
            params={'temperature': 0.7},
        ),
    ],
    name='synth_default',
    synthesizer=sf.Model(
        'gemini-cli/gemini-3.1-flash-lite',
    ),
)

report = sf.evaluate(
    candidate,
    benchmark='ifeval',
    limit=50,
)