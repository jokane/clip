# This is an illustration of the need for memoization when computing audio.
#
# Without memoization this took about a minute on my computer, mostly from
# generating the sine wave 100 times.  With memoization ---which, when you read
# this, will be implemented throughout the library--- it takes about 5 seconds.
#
# It's tough to make this a proper test because it's a performance difference,
# not really a behavior difference.  But good to see confirmation that it's
# working.
#


from clip import *

L = 100

sine = sine_wave(frequency=440,
                 volume=0.5,
                 length=L,
                 sample_rate=44000,
                 num_channels=2)

chained = chain([ slice_clip(sine, t, t+1) for t in range(L)])
 
save_mp4(chained, '/dev/null', frame_rate=10)
