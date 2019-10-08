# FFT using R

x <- 1:2^30
elapsedTime = system.time({
    fft(fft(x), inverse = TRUE)/length(x)
})[3]
print(sprintf("Elapsed Time: %3.3f ms", elapsedTime))