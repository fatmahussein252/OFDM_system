clear all;
%close all;


Eb=1;
SNR_vector= -20:20;
% Define positions on signal space
QPSK_positions=[-1-1i -1+1i 1-1i 1+1i];
QAM16_positions=[-3-3i -3-1i -3+3i -3+1i -1-3i -1-1i -1+3i -1+1i 3-3i 3-1i 3+3i 3+1i 1-3i 1-1i 1+3i 1+1i];
% 

                %---------------QPSK-----------------%
%%----------------Transmitter------------%%
Interleaver_size_QPSK = 256; 
num_bits_QPSK = Interleaver_size_QPSK * 500;
% Generate random bits 
Data_QPSK = randi([0 1] , 1 , num_bits_QPSK);
% Apply interleaver Block
Interleaved_Data_QPSK = interleave(Data_QPSK,num_bits_QPSK,Interleaver_size_QPSK,16,16);
% Apply mapper Block
bits_per_symbol_QPSK = 2;
mapper_out_QPSK = QPSK_map(Interleaved_Data_QPSK,num_bits_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
IFFT_out_QPSK = Apply_ifft(mapper_out_QPSK,num_bits_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
prefix_QPSK = 32;
Tx_out_QPSK = cyc_prefix(IFFT_out_QPSK,num_bits_QPSK,prefix_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
%%---------------channel-----------------%%
[Rx_signal_QPSK,h_channel_QPSK] = channel(Tx_out_QPSK,num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,Eb,SNR_vector,bits_per_symbol_QPSK,'Rayleigh');
[Rx_signal_QPSK_FS,h_channel_QPSK_FS] = channel(Tx_out_QPSK,num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,Eb,SNR_vector,bits_per_symbol_QPSK,'Frequency selective');
%%---------------Reciever----------------%%
removed_channel_effect_QPSK = rm_channel(Rx_signal_QPSK_FS,SNR_vector,h_channel_QPSK_FS);
removed_prefix_data_QPSK = rm_prefix(removed_channel_effect_QPSK,num_bits_QPSK,prefix_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK,SNR_vector);
FFT_out_QPSK = Apply_fft(removed_prefix_data_QPSK,num_bits_QPSK,Interleaver_size_QPSK,SNR_vector,bits_per_symbol_QPSK);
estimated_signal_QPSK = estimate(FFT_out_QPSK,num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,SNR_vector,bits_per_symbol_QPSK,'QPSK');
demapper_out_QPSK = Demap(estimated_signal_QPSK,num_bits_QPSK,bits_per_symbol_QPSK,Interleaver_size_QPSK,SNR_vector,QPSK_positions);
deinterleaved_data_QPSK = deinterleave(Data_QPSK,demapper_out_QPSK, num_bits_QPSK, Interleaver_size_QPSK,SNR_vector,16,16);
BER_QPSK = getBER(Data_QPSK,deinterleaved_data_QPSK,num_bits_QPSK,SNR_vector,1);


%------------------QPSK 3 bit repition code-------------------%
%%----------------Transmitter------------%% 
% repeat generated random bits 
Data_QPSK = repelem(Data_QPSK, 3);
% Apply interleaver Block
Interleaved_Data_QPSK = interleave(Data_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,16,16);
% Apply mapper Block
bits_per_symbol_QPSK = 2;
mapper_out_QPSK = QPSK_map(Interleaved_Data_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
IFFT_out_QPSK = Apply_ifft(mapper_out_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
prefix_QPSK = 32;
Tx_out_QPSK = cyc_prefix(IFFT_out_QPSK,3*num_bits_QPSK,prefix_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK);
%%---------------channel-----------------%%
[Rx_signal_QPSK,h_channel_QPSK] = channel(Tx_out_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,Eb,SNR_vector,bits_per_symbol_QPSK,'Rayleigh');
[Rx_signal_QPSK_FS,h_channel_QPSK_FS] = channel(Tx_out_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,Eb,SNR_vector,bits_per_symbol_QPSK,'Frequency selective');
%%---------------Reciever----------------%%
removed_channel_effect_QPSK = rm_channel(Rx_signal_QPSK_FS,SNR_vector,h_channel_QPSK_FS);
removed_prefix_data_QPSK = rm_prefix(removed_channel_effect_QPSK,3*num_bits_QPSK,prefix_QPSK,Interleaver_size_QPSK,bits_per_symbol_QPSK,SNR_vector);
FFT_out_QPSK = Apply_fft(removed_prefix_data_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,SNR_vector,bits_per_symbol_QPSK);
estimated_signal_QPSK = estimate(FFT_out_QPSK,3*num_bits_QPSK,Interleaver_size_QPSK,prefix_QPSK,SNR_vector,bits_per_symbol_QPSK,'QPSK');
demapper_out_QPSK = Demap(estimated_signal_QPSK,3*num_bits_QPSK,bits_per_symbol_QPSK,Interleaver_size_QPSK,SNR_vector,QPSK_positions);
deinterleaved_data_QPSK_repeated = deinterleave(Data_QPSK,demapper_out_QPSK, 3*num_bits_QPSK, Interleaver_size_QPSK,SNR_vector,16,16);
BER_QPSK_repeated = getBER(Data_QPSK,deinterleaved_data_QPSK_repeated,3*num_bits_QPSK,SNR_vector,3);

%%%%%%%plot Both%%%%%%%%
plot_BER(SNR_vector,BER_QPSK,BER_QPSK_repeated,'BER for Rayleigh channel for uncoded and 3 bit coded signal - QPSK');




                %---------------16_QAM-----------------%
%%----------------Transmitter------------%%
Interleaver_size_16QAM = 512; 
num_bits_16QAM = Interleaver_size_16QAM * 500;
% Generate random bits 
Data_QAM = randi([0 1] , 1 , num_bits_16QAM);
% Apply interleaver Block
Interleaved_Data_16QAM = interleave(Data_QAM,num_bits_16QAM,Interleaver_size_16QAM,32,16);
% Apply mapper Block
bits_per_symbol_16QAM = 4;
mapper_out_16QAM = QAM_map(Interleaved_Data_16QAM,num_bits_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM,QAM16_positions);
IFFT_out_16QAM = Apply_ifft(mapper_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM);
prefix_16QAM = 32;
Tx_out_16QAM = cyc_prefix(IFFT_out_16QAM,num_bits_16QAM,prefix_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM);
%%---------------channel-----------------%%
[Rx_signal_16QAM,h_channel_16QAM] = channel(Tx_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,Eb,SNR_vector,bits_per_symbol_16QAM,'Rayleigh');
[Rx_signal_16QAM_FS,h_channel_16QAM_FS] = channel(Tx_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,Eb,SNR_vector,bits_per_symbol_16QAM,'Frequency selective');
%%---------------Reciever----------------%%
removed_channel_effect_16QAM = rm_channel(Rx_signal_16QAM_FS,SNR_vector,h_channel_16QAM_FS);
removed_prefix_data_16QAM = rm_prefix(removed_channel_effect_16QAM,num_bits_16QAM,prefix_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM,SNR_vector);
FFT_out_16QAM = Apply_fft(removed_prefix_data_16QAM,num_bits_16QAM,Interleaver_size_16QAM,SNR_vector,bits_per_symbol_16QAM);
estimated_signal_16QAM = estimate(FFT_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,SNR_vector,bits_per_symbol_16QAM,'16_QAM');
demapper_out_16QAM = Demap(estimated_signal_16QAM,num_bits_16QAM,bits_per_symbol_16QAM,Interleaver_size_16QAM,SNR_vector,QAM16_positions);
deinterleaved_data_16QAM = deinterleave(Data_QAM,demapper_out_16QAM, num_bits_16QAM, Interleaver_size_16QAM,SNR_vector,16,32,'BER of 16_QAM uncoded');
BER_16QAM = getBER(Data_QAM,deinterleaved_data_16QAM,num_bits_16QAM,SNR_vector,1);
%------------------16QAM 3 bit repition code-------------------%
%%----------------Transmitter------------%%
% Repeat generated random bits 
Data_QAM = repelem(Data_QAM,3);
% Apply interleaver Block
Interleaved_Data_16QAM = interleave(Data_QAM,3*num_bits_16QAM,Interleaver_size_16QAM,32,16);
% Apply mapper Block
bits_per_symbol_16QAM = 4;
mapper_out_16QAM = QAM_map(Interleaved_Data_16QAM,3*num_bits_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM,QAM16_positions);
IFFT_out_16QAM = Apply_ifft(mapper_out_16QAM,3*num_bits_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM);
prefix_16QAM = 32;
Tx_out_16QAM = cyc_prefix(IFFT_out_16QAM,3*num_bits_16QAM,prefix_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM);
%%---------------channel-----------------%%
[Rx_signal_16QAM,h_channel_16QAM] = channel(Tx_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,Eb,SNR_vector,bits_per_symbol_16QAM,'Rayleigh');
[Rx_signal_16QAM_FS,h_channel_16QAM_FS] = channel(Tx_out_16QAM,num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,Eb,SNR_vector,bits_per_symbol_16QAM,'Frequency selective');
%%---------------Reciever----------------%%
removed_channel_effect_16QAM = rm_channel(Rx_signal_16QAM_FS,SNR_vector,h_channel_16QAM_FS);
removed_prefix_data_16QAM = rm_prefix(removed_channel_effect_16QAM,3*num_bits_16QAM,prefix_16QAM,Interleaver_size_16QAM,bits_per_symbol_16QAM,SNR_vector);
FFT_out_16QAM = Apply_fft(removed_prefix_data_16QAM,3*num_bits_16QAM,Interleaver_size_16QAM,SNR_vector,bits_per_symbol_16QAM);
estimated_signal_16QAM = estimate(FFT_out_16QAM,3*num_bits_16QAM,Interleaver_size_16QAM,prefix_16QAM,SNR_vector,bits_per_symbol_16QAM,'16_QAM');
demapper_out_16QAM = Demap(estimated_signal_16QAM,3*num_bits_16QAM,bits_per_symbol_16QAM,Interleaver_size_16QAM,SNR_vector,QAM16_positions);
deinterleaved_data_16QAM_repeated = deinterleave(Data_QAM,demapper_out_16QAM, 3*num_bits_16QAM, Interleaver_size_16QAM,SNR_vector,16,32,'BER of 16_QAM uncoded');
BER_16QAM_repeated = getBER(Data_QAM,deinterleaved_data_16QAM_repeated,3*num_bits_16QAM,SNR_vector,3);

%%%%%%%%plot Both%%%%%%%%
plot_BER(SNR_vector,BER_16QAM,BER_16QAM_repeated,'frequency selective channel for uncoded and 3 bit coded signal - 16QAM');


%-----------------------------Tx Blocks functions-----------------------------%
% Interleave the data to resolve deep faded symbols correctly 
function Interleaved_Data = interleave(Data, num_bits, Interleaver_size,r,c)
    % Pre-allocate the output matrix
    Interleaved_Data = zeros(num_bits / Interleaver_size, Interleaver_size);
    Interleaved_matrix_trans = zeros(c,r,num_bits/Interleaver_size);
    % Reshape the data into 16x16 blocks
    Interleaved_matrix = reshape(Data, r, c, []);
    
    % Perform interleaving (row-column transposition)
    for i = 1:num_bits / Interleaver_size
        Interleaved_matrix_trans(:, :, i) = Interleaved_matrix(:, :, i).';
        Interleaved_Data(i, :) = reshape(Interleaved_matrix(:, :, i), [1, Interleaver_size]);
    end
end
% the mapper function of the QPSK
function mapper_out=QPSK_map(Interleaved_Data,num_bits,Interleaver_size,bits_per_symbol)
    mapper_out = zeros(num_bits/Interleaver_size,Interleaver_size/bits_per_symbol);
    for p = 1: num_bits/Interleaver_size
      m=1;
      for i=1:2:Interleaver_size-2
          if (Interleaved_Data(p,i) == 0 && Interleaved_Data(p,i+1) == 0)
              mapper_out(p,m)= -1-1i;
          elseif (Interleaved_Data(p,i) == 0 && Interleaved_Data(p,i+1) == 1)
              mapper_out(p,m)= -1+1i;
          elseif (Interleaved_Data(p,i) == 1 && Interleaved_Data(p,i+1) == 0)
              mapper_out(p,m)= 1-1i;
          elseif (Interleaved_Data(p,i) == 1 && Interleaved_Data(p,i+1) == 1)
              mapper_out(p,m)= 1+1i;
          end
          m = m+1 ;
      end
    end
end
% The mapper function of the 16_QAM
function mapper_out_16QAM=QAM_map(Interleaved_Data,num_bits,Interleaver_size,bits_per_symbol,positions)
    mapper_out_16QAM = zeros(num_bits/Interleaver_size,Interleaver_size/bits_per_symbol);
    num_symbols = Interleaver_size/bits_per_symbol;
    for p = 1:num_bits/Interleaver_size
        mapper_vector_bin=reshape(Interleaved_Data(p,:),bits_per_symbol,Interleaver_size/bits_per_symbol).';
        for s=1:num_symbols
            mapper_vector_Dec(s) = bin2dec(num2str(mapper_vector_bin(s,:)));
            mapper_out_16QAM(p,s)=positions(mapper_vector_Dec(s)+1);
        end
    end
end
% Apply ifft to generate orthogonal carriers
function IFFT_out = Apply_ifft(mapper_out,num_bits,Interleaver_size,bits_per_symbol)
    ifft_size = Interleaver_size/bits_per_symbol;
    for i = 1:num_bits/Interleaver_size
        IFFT_out(i,:) = ifft(mapper_out(i,:),ifft_size);
    end
end
% Add prefix part to resolve symbols correctly
function Tx_out = cyc_prefix(IFFT_out,num_bits,prefix,Interleaver_size,bits_per_symbol)
    ifft_size = Interleaver_size/bits_per_symbol;
    num_symbols = (num_bits / Interleaver_size) *(ifft_size+prefix);
    for i = 1:num_bits/Interleaver_size
        prefix_part = IFFT_out(i,ifft_size-prefix+1:ifft_size);
        Tx_Data(i,:) = cat(2,prefix_part,IFFT_out(i,:));
    end
    Tx_Data = Tx_Data.';
    Tx_out = reshape(Tx_Data,1,num_symbols);
    
end
% the Rayleigh and frequency selective channels
function [Rx_signal,h_channel] = channel(Tx_out,num_bits,Interleaver_size,prefix,Eb,SNR_vector,bits_per_symbol,channel_type)
%------1) Generate AWGN---------%
% Generate a unity variance, zero mean additive white Gaussian noise signal
% with the same size as transmitted signal.
sym_num = size(Tx_out,2);
I_noise=randn(1,sym_num);
Q_noise=randn(1,sym_num);
%Get the Noisy signal for each SNR value
Eavg=sum(abs(Tx_out).^2,2)/sym_num;

if strcmp(channel_type,'Rayleigh')
    % Generate Rayleigh fading channel
    h_channel = sqrt(1/2) * (randn(1) + 1j * randn(1));
    h_channel = repelem(h_channel,sym_num);
elseif strcmp(channel_type,'Frequency selective')
    % Generate frequency selective fading channel
%     sub_ch = 10;
%     h_channel = sqrt(1/2) * (randn(1,sub_ch) + 1j * randn(1,sub_ch));
%     h_channel = repelem(h_channel,sym_num/sub_ch);
%     fft_size = size(Tx_out,2)/bits_per_symbol;
%     for p = 1:num_bits/Interleaver_size
%         start_idx = (p - 1) * fft_size + 1;
%         end_idx = start_idx + fft_size - 1;
%         Tx_freq_frames(p,:) = fft(Tx_out(start_idx:end_idx),fft_size);
%     end
%     Tx_freq = reshape(Tx_freq_frames,1,size(Tx_out,2));
%     channel_response = h_channel .* Tx_freq;
%     for p = 1:num_bits/Interleaver_size
%         start_idx = (p - 1) * fft_size + 1;
%         end_idx = start_idx + fft_size - 1;
%         Tx_time_frames(p,:) = ifft(Tx_out(start_idx:end_idx),fft_size);
%     end
%     Tx_time = reshape(Tx_time_frames,1,size(Tx_out,2));
sub_ch = 10;

% Generate channel coefficients and repeat them for all symbols
h_channel = sqrt(1/2) * (randn(1, sub_ch) + 1j * randn(1, sub_ch));
h_channel = repelem(h_channel, size(Tx_out, 2) / sub_ch);

% FFT size
fft_size = size(Tx_out, 2) / bits_per_symbol;

% Number of blocks
num_blocks = floor(size(Tx_out, 2) / fft_size); % Ensure we only process full blocks

% Preallocate matrices
Tx_freq_frames = zeros(num_blocks, fft_size);
Tx_time_frames = zeros(num_blocks, fft_size);

% Perform FFT on each block of size fft_size
for p = 1:num_blocks
    start_idx = (p - 1) * fft_size + 1;
    end_idx = start_idx + fft_size - 1;
    Tx_freq_frames(p, :) = fft(Tx_out(start_idx:end_idx), fft_size);
end

% Reshape to 1D frequency domain signal
Tx_freq = reshape(Tx_freq_frames.', 1, []);

% Apply channel response in the frequency domain
channel_response = h_channel(1:numel(Tx_freq)) .* Tx_freq;

% Perform IFFT on each block of size fft_size
for p = 1:num_blocks
    start_idx = (p - 1) * fft_size + 1;
    end_idx = start_idx + fft_size - 1;
    Tx_time_frames(p, :) = ifft(channel_response(start_idx:end_idx), fft_size);
end

% Reshape to 1D time domain signal
Tx_time = reshape(Tx_time_frames.', 1, []);

end
for j=1:length(SNR_vector)
    %calculate variance from SNR (SNR=Es/Ns)
    No_vector(j)=Eb / 10^(SNR_vector(j)/10);
    variance_vector(j)=(No_vector(j)*(Eavg/bits_per_symbol))/2; %variance=Ns/2
    %Scale the noise sequence to have variance = N0/2 by multiplying the sequence
    %by sqrt(N0/2).
    I_scaled_noise=sqrt(variance_vector(j)) * I_noise;
    Q_scaled_noise=sqrt(variance_vector(j)) * Q_noise;
    AWGN = I_scaled_noise + Q_scaled_noise *1i;
    %Add the noise to the transmitted sequence
    if strcmp(channel_type,'Rayleigh')
        Rx_signal(j,:)=h_channel .* Tx_out + AWGN;
    elseif strcmp(channel_type,'Frequency selective')
        Rx_signal(j,:)=Tx_time + AWGN;
    end
end

end
%-----------------------------Rx Blocks functions-----------------------------%
% Remove the channel effect
function removed_channel_effect = rm_channel(Rx_signal,SNR_vector,h_channel)
removed_channel_effect = zeros(size(Rx_signal));
for i=1:length(SNR_vector)
    removed_channel_effect(i,:) = Rx_signal(i,:) ./ h_channel;
end
end
% remove the prefix part from the symbols
function removed_prefix_data = rm_prefix(removed_channel_effect,num_bits,prefix,Interleaver_size,bits_per_symbol,SNR_vector)
    removed_prefix_data = [];
    ifft_size = Interleaver_size/bits_per_symbol;
   
    for p = 1:ifft_size+prefix:size(removed_channel_effect,2)
        removed_prefix_symbol = removed_channel_effect(:,p+prefix:p+prefix+ifft_size-1);
        removed_prefix_data = cat(2,removed_prefix_data,removed_prefix_symbol);
    end
    
end
% Apply fft
function FFT_out = Apply_fft(removed_prefix_data,num_bits,Interleaver_size,SNR_vector,bits_per_symbol)
for i=1:length(SNR_vector)
    FFT_cat =[];
    fft_size = Interleaver_size/bits_per_symbol;
    for p = 1:fft_size:size(removed_prefix_data,2)
        FFT_Data = fft(removed_prefix_data(i,p:fft_size+p-1),fft_size);
        FFT_cat = [FFT_cat,  FFT_Data];
    end
    FFT_out(i,:) = FFT_cat;
end
end
% Estimate the recieved coded and uncoded signals
function estimated_signal = estimate(FFT_out,num_bits,Interleaver_size,prefix,SNR_vector,bits_per_symbol,mod_type)
estimated_signal =complex(zeros(size(FFT_out)));
for i=1:length(SNR_vector)
    %Estimation of symbols
    if strcmp(mod_type,'QPSK')
        for s=1:size(FFT_out,2)
            if real(FFT_out(i,s))>=0 && imag(FFT_out(i,s))>=0
                estimated_signal(i,s)=1+1i;
            elseif real(FFT_out(i,s))>=0 && imag(FFT_out(i,s))<0
                estimated_signal(i,s)=1-1i;
            elseif real(FFT_out(i,s))<0 && imag(FFT_out(i,s))>=0
                estimated_signal(i,s)=-1+1i;
            else
                estimated_signal(i,s)=-1-1i;
            end
        end
        
    elseif strcmp(mod_type,'16_QAM')
        for s=1:size(FFT_out,2)
            if real(FFT_out(i,s))>0
                if real(FFT_out(i,s))>2
                    estimated_signal(i,s)=3+imag(FFT_out(i,s))*1i;
                else
                    estimated_signal(i,s)=1+imag(FFT_out(i,s))*1i;
                end
            else
                if real(FFT_out(i,s))<-2
                    estimated_signal(i,s)=-3+imag(FFT_out(i,s))*1i;
                else
                    estimated_signal(i,s)=-1+imag(FFT_out(i,s))*1i;
                end
            end
            if imag(estimated_signal(i,s))>0
                if imag(estimated_signal(i,s))>2
                    estimated_signal(i,s)=real(estimated_signal(i,s))+3i;
                else
                    estimated_signal(i,s)=real(estimated_signal(i,s))+1i;
                end
            else
                if imag(estimated_signal(i,s))<-2
                    estimated_signal(i,s)=real(estimated_signal(i,s))-3i;
                else
                    estimated_signal(i,s)=real(estimated_signal(i,s))-1i;
                end
            end
        end
    end
end
end
% Demap recieved signal to bits
function demapper_out = Demap(estimated_signal,num_bits,bits_per_symbol,Interleaver_size,SNR_vector,positions)
%Demapping estimated symbols to the corresponding decimal values then to
%binary bits
for i=1:length(SNR_vector)
    for s=1:size(estimated_signal,2)
        for p=1:length(positions)
            if estimated_signal(i,s) == positions(p)
                demapper_vector_Dec(i,s)=p-1;
            end
        end
    end
    demapper_vector_str=dec2bin(demapper_vector_Dec(i,:),2);
    for b=1:bits_per_symbol
        demapper_vector_bin(:,b)=str2num(demapper_vector_str(:,b));
    end
    demapper_out(i,:) = reshape(demapper_vector_bin',1,size(estimated_signal,2)*bits_per_symbol);
    
   
end
end
% deinterleave the data
function deinterleaved_out = deinterleave(Data, demapper_out, num_bits, Interleaver_size, SNR_vector,r,c,str)
    % Pre-allocate the deinterleaved output
    deinterleaved_out = zeros(length(SNR_vector), num_bits);
    BER = zeros(length(SNR_vector), 1); % Bit error rate for each SNR
    Interleaved_matrix_trans = zeros(c,r,num_bits/Interleaver_size);
    % Loop through each SNR value
    for i = 1:length(SNR_vector)
        % Reshape the demapper output back into 16x16 blocks
        Interleaved_matrix = reshape(demapper_out(i, :), r, c, []);

        % Perform deinterleaving (column-row transposition)
        for m = 1:num_bits / Interleaver_size
            Interleaved_matrix_trans(:, :, m) = Interleaved_matrix(:, :, m).';
            start_idx = (m - 1) * Interleaver_size + 1;
            end_idx = start_idx + Interleaver_size - 1;
            deinterleaved_out(i, start_idx:end_idx) = reshape(Interleaved_matrix(:, :, m), [1, Interleaver_size]);
        end
    end
end
% get BER
function BER = getBER(Data,deinterleaved_out,num_bits,SNR_vector,code_type)
if code_type == 1
    for i = 1:length(SNR_vector)
        % Calculate the bit error rate (BER) for each SNR
        error_counter = sum(deinterleaved_out(i, :) ~= Data);
        BER(i) = error_counter / num_bits;
    end
elseif code_type == 3
    Repeated_signal = zeros(length(SNR_vector),num_bits);
    Non_Repeated_signal = zeros(length(SNR_vector),num_bits/3);
    for i = 1:length(SNR_vector)
        for c=1:3:num_bits-3
            if (deinterleaved_out(i,c)+deinterleaved_out(i,c+1)+deinterleaved_out(i,c+2))>=2
                Repeated_signal(i,c)=1;
                Repeated_signal(i,c+1)=1;
                Repeated_signal(i,c+2)=1;
            else
                Repeated_signal(i,c)=0;
                Repeated_signal(i,c+1)=0;
                Repeated_signal(i,c+2)=0;
            end
        end
    Non_Repeated_signal(i,:) = Repeated_signal(i,1:3:end);
    non_repeated_data = Data(1:3:end);
    % Calculate the bit error rate (BER) for each SNR
    error_counter = sum(Non_Repeated_signal(i, :) ~= non_repeated_data);
    BER(i) = (error_counter * 3)/ num_bits;
    end
end
end
% plot coded vs uncoded BER
function plot_BER(SNR_vector,BER,BER_repeated,str)
    figure;
    semilogy(SNR_vector,BER);
    hold on;
    semilogy(SNR_vector,BER_repeated);
    ylim([0.35 0.5]);
    xlim([-20 5]);
    xlabel('Eb/No');
    ylabel('BER');
    title([str]);
    legend("BER uncoded","BER 3 bit coded");
    hold off;
end


