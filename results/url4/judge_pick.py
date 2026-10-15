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
    name='judge_pick',
    synthesizer=sf.Judge('gemini-cli/gemini-3.1-flash-lite', returns='label', rubric='Pick the answer that best follows every instruction in the request.'),
)

report = sf.evaluate(
    candidate,
    benchmark='ifeval',
    limit=50,
)