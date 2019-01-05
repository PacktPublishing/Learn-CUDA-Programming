# FFT

num_sample = 8192
x = single(rand(num_sample));
n_fft = 2^nextpow2(num_sample);

start = clock();
y = fft(x, n_fft);
ix = ifft(y, n_fft);
elapsedTime = etime(clock(), start);

printf("Elapsed Time: %.3f ms\n", elapsedTime);
