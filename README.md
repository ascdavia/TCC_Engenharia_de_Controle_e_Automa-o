<h1 align="center"> COMPARAÇÃO DE DESEMPENHO NA CLASSIFICAÇÃO DE BIOSINAIS COM DEEP LEARNING: UMA ABORDAGEM UTILIZANDO A TRANSFORMADA DE WAVELET COMO PRÉ-PROCESSAMENTO DE DADOS </h1>

Sinais biológicos ou biosinais, são sinais provenientes das atividades existentes dentro de um ser vivo. A analise desses sinais permite entender melhor o funcionamento de diversos parâmetros. Nesse sentido, esse trabalho visa realizar a classificação de biosinais provenientes de seis gestos de mão que são utilizados em processos de reabilitação motora.

![grasps_en](https://user-images.githubusercontent.com/76635621/182120286-b042691d-41a7-46e7-a4e4-ab10f56e0023.PNG)

## :small_blue_diamond: Redes Utilizadas

- `Convolutional Neural Networks (CNN)`
- `Long Short Term Memory (LSTM)`
- `Ensemble utilizando CNN e LSTM`


## :small_blue_diamond: Códigos

- `CNN`: Testes utilizando CNN.

- `Ensemble`: Testes utilizando o ensemble das redes CNN e LSTM.

- `LSTM`: Testes utilizando LSTM.

- `Criação_DF_Movimentos_e_MatrizX.ipynb`: Como os arquivos da base de dados foram separados por movimento e por pessoa, foi necessário juntar eles formando arquivos .csv para cada movimento e um arquivo para juntar todos os movimentos.

- `Plot_dos_Sinais.ipynb`: Plotagem dos sinais para ter uma visualização de como eles são além de comparar entre si os sinais gerados pelo homem e pela mulher.


## :small_blue_diamond: Database

Os sinais presentes na base de dados são frutos de um teste realizado com cinco pessoas, sendo dois homens e três mulheres, onde as mesmas precisaram executar os seis movimentos de mão, sendo que cada movimento foi executado durante seis segundos por trinta rounds de repetição. A captação do sinal foi feita por dois sensores EMG, esse sinal passava por um conversor A/D NI USB-6009 com uma taxa de amostragem de 500 Hz e enviado a um computado que fez o registro do sinal e posteriormente salvos em arquivos .m (MATLAB). 

Por uma questão de praticidade, foi utilizado o MATLAB para exportar a base da dados em arquivos .csv.

- `sEMG_Basic_Hand_Movements_Database_1`: Arquivos originais em .m.

- `sEMG_Basic_Hand_Movements_Database_2`: Arquivos originais em .m.

- `sEMG_Basic_Hand_movements_upatras_csv_files`: Arquivos exportados em .csv.

- `sEMG_Basic_Hand_movements_upatras.zip`: Arquivos originais compactados.
