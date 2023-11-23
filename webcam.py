from noddingpigeon.inference import predict_video

while True:
    result = predict_video()
    print(result)
    # Example result:
    # {'gesture': 'nodding',
    #  'probabilities': {'has_motion': 1.0,
    #   'gestures': {'nodding': 0.9576354622840881,
    #    'turning': 0.042364541441202164}}}

