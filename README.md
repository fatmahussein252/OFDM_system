# OFDM_system
matlab simulation of sender and reciever in OFDM system  for wireless digital communications
## OFDM system Block diagram

![image](https://github.com/user-attachments/assets/231aa8dc-8ba6-4952-8a31-b1a58268edb6)

The above blocks were designed as follows:
1. Coding: No coding and rate 1/3 repetition code were used. Generating number of bits = 500 * interleaver size for each modulation before repetition.
2. Interleaver for two modulation types:
- For QPSK, the size of the interleaver used is 16 by 16.
- For 16QAM, the interleaver size used is 32 by 16.
3. Mapper: The mappers used are the same as those in the single carrier system
4. IFFT: used a size 128 IFFT block. So the data were divided into blocks of 128 symbol before it. 
5. Channel:Two channel models were considered:
- Rayleigh flat fading channel: Same as single carrier system
- Frequency selective Fading channel: dividing data into 10 subchannels and assuming that every sub-channel is independently faded by a different Rayleigh fading channel.  The fading was modeled in the frequency domain.
- The SNR range used: -20:20 dB
6. Receiver: A receiver designed to receive the signal described above in the two cases of the channel model. All the Blocks described above were designed with reversed operations.
## outputs:
1. the BER of Rayleigh fading channel for 16_QAM modulation

![image](https://github.com/user-attachments/assets/df5f2842-33cc-4e92-8c28-d294caffc883)

2. the BER of frequency selective fading channel for 16_QAM modulation

![image](https://github.com/user-attachments/assets/a57988f8-e472-4713-b019-7a180941f6e2)


