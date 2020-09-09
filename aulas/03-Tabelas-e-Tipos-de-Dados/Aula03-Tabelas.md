
# Aula 03 - Tabelas e Tipos de Dados

## Introdução

Neste notebook vamos explorar um pouco de dados tabulares. A principal biblioteca para leitura de dados tabulares em Python se chama **pandas**. A mesma é bastante poderosa implementando uma série de operações de bancos de dados (e.g., groupby e join). Nossa discussão será focada em algumas das funções principais do pandas que vamos explorar no curso. Existe uma série ampla de funcionalidades que a biblioteca (além de outras) vai trazer. 

Caso necessite de algo além da aula, busque na documentação da biblioteca. Por fim, durante esta aula, também vamos aprender um pouco de bash.

## Objetivos

1. Aprender Pandas
2. Entender diferentes tipos de dados
3. Básico de filtros e seleções

## Resultado Esperado

1. Aplicação de filtros básicos para gerar insights nos dados de dados tabulares

## Imports básicos

A maioria dos nossos notebooks vai iniciar com os imports abaixo.
1. pandas: dados tabulates
1. matplotlib: gráficos e plots

A chamada `plt.ion` habilita gráficos do matplotlib no notebook diretamente. Caso necesse salvar alguma figura, chame `plt.savefig` após seu plot.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
```

## Series

Existem dois tipos base de dados em pandas. O primeiro, Series, representa uma coluna de dados. Um combinação de Series vira um DataFrame (mais abaixo). Diferente de um vetor `numpy`, a Series de panda captura uma coluna de dados (ou vetor) indexado. Isto é, podemos nomear cada um dos valores.


```python
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
```


```python
data
```




    a    0.25
    b    0.50
    c    0.75
    d    1.00
    dtype: float64



Note que podemos usar como um vetor


```python
data[0]
```




    0.25



Porém o índice nos ajuda. Para um exemplo trivial como este não será tão interessante, mas vamos usar o mesmo.


```python
data.index
```




    Index(['a', 'b', 'c', 'd'], dtype='object')



Com .loc acessamos uma linha do índice com base no nome. Então:

1. `series.loc[objeto_python]` - valor com o devido nome.
1. `series.iloc[int]` - i-ésimo elemento da Series.


```python
data.loc['a']
```




    0.25




```python
data.loc['b']
```




    0.5



Com `iloc` acessamos por número da linha, estilho um vetor.


```python
data.iloc[0]
```




    0.25




```python
data[0]
```




    0.25



## Data Frames

Ao combinar várias Series com um índice comum, criamos um **DataFrame**. Não é tão comum gerar os mesmos na mão como estamos fazendo, geralmente carregamos DataFrames de arquivos `.csv`, `.json` ou até de sistemas de bancos de dados `mariadb`. De qualquer forma, use os exemplos abaixo para entender a estrutura de um dataframe.

Lembre-se que {}/dict é um dicionário (ou mapa) em Python. Podemos criar uma série a partir de um dicionário
index->value


```python
area_dict = {'California': 423967,
             'Texas': 695662,
             'New York': 141297,
             'Florida': 170312,
             'Illinois': 149995}
```

A linha abaixo pega todas as chaves.


```python
list(area_dict.keys())
```




    ['California', 'Texas', 'New York', 'Florida', 'Illinois']



Agora todas as colunas


```python
list(area_dict.values())
```




    [423967, 695662, 141297, 170312, 149995]



Acessando um valor.


```python
area_dict['California']
```




    423967



Podemos criar a série a partir do dicionário, cada chave vira um elemento do índice. Os valores viram os dados do vetor.


```python
area = pd.Series(area_dict)
area
```




    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    dtype: int64



Agora, vamos criar outro dicionário com a população dos estados.


```python
pop_dict = {'California': 38332521,
            'Texas': 26448193,
            'New York': 19651127,
            'Florida': 19552860,
            'Illinois': 12882135}
pop = pd.Series(pop_dict)
pop
```




    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    Illinois      12882135
    dtype: int64



Por fim, observe que o DataFrame é uma combinação de Series. Cada uma das Series vira uma coluna da tabela de dados.


```python
data = pd.DataFrame({'area':area, 'pop':pop})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>



Agora o use de `.loc e .iloc` deve ficar mais claro, observe os exemplos abaixo.


```python
data.loc['California']
```




    area      423967
    pop     38332521
    Name: California, dtype: int64




```python
data.loc[['California', 'Texas']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
  </tbody>
</table>
</div>



Note que o uso de `iloc` retorna a i-ésima linha. O problema é que nem sempre nos dataframes esta ordem vai fazer sentido. O `iloc` acaba sendo mais interessante para iteração (e.g., passar por todas as linhas.)


```python
data.iloc[0]
```




    area      423967
    pop     38332521
    Name: California, dtype: int64



## Slicing

Agora, podemos realizar slicing no DataFrame. Slicing é uma operação Python que retorna sub-listas/sub-vetores. Caso não conheça, tente executar o exemplo abaixo:

```python
l = []
l = [7, 1, 3, 5, 9]
print(l[0])
print(l[1])
print(l[2])

# Agora, l[bg:ed] retorna uma sublista iniciando em bg e terminando em ed-1
print(l[1:4])
```


```python
l = []
l = [7, 1, 3, 5, 9]
print(l[0])
print(l[1])
print(l[2])

# Agora, l[bg:ed] retorna uma sublista iniciando em bg e terminando em ed-1
print(l[1:4])
```

    7
    1
    3
    [1, 3, 5]


Voltando para o nosso **dataframe**, podemos realizar o slicing usando o `.iloc`.


```python
data.iloc[2:4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
  </tbody>
</table>
</div>



## Modificando DataFrames

Series e DataFrames são objetos mutáveis em Python. Podemos adicionar novas colunas em DataFrama facilmente da mesma forma que adicionamos novos valores em um mapa. Por fim, podemos também mudar o valor de linhas específicas e adicionar novas linhas.


```python
data['density'] = data['pop'] / data['area']
data.loc['Texas']
```




    area       6.956620e+05
    pop        2.644819e+07
    density    3.801874e+01
    Name: Texas, dtype: float64




```python
df = data
```


```python
df.index
```




    Index(['California', 'Texas', 'New York', 'Florida', 'Illinois'], dtype='object')



## Arquivos

Antes de explorar DataFrames em arquivos, vamos ver como um notebook na é um shell bastante poderoso. Ao usar uma exclamação (!) no notebook Jupyter, conseguimos executar comandos do shell do sistema. Em particular, aqui estamos executando o comando ls para indentificar os dados da pasta atual.

Tudo que executamos com `!` é um comando do terminal do unix. Então, este notebook só deve executar as linhas abaixo em um `Mac` ou `Linux`.


```python
!ls .
```

    Aula03-Tabelas.ipynb  movie_metadata.csv_movie_metadata.csv
    baby.csv	      nba_salaries.csv


Com a opção -lha, mostramos meta-dados dos arquivos como o owner, tamanho e permissões. Note que todos os arquivos são .csv, isto é comma separated.


```python
!ls -lha .
```

    total 150M
    drwxr-xr-x  3 flaviovdf flaviovdf 4.0K Sep  9 12:16 .
    drwxr-xr-x 25 flaviovdf flaviovdf 4.0K Sep  9 12:20 ..
    -rw-r--r--  1 flaviovdf flaviovdf 142K Sep  9 12:21 Aula03-Tabelas.ipynb
    -rw-r--r--  1 flaviovdf flaviovdf 148M Mar 11 14:16 baby.csv
    drwxr-xr-x  2 flaviovdf flaviovdf 4.0K Sep  9 11:52 .ipynb_checkpoints
    -rw-r--r--  1 flaviovdf flaviovdf 1.5M Mar 11 14:15 movie_metadata.csv_movie_metadata.csv
    -rw-r--r--  1 flaviovdf flaviovdf  18K Mar 11 14:15 nba_salaries.csv


Vamos identificar qual a cara de um csv. O programa `head` imprime as primeiras `n` linhas de um arquivo.


```python
!head baby.csv
```

    Id,Name,Year,Gender,State,Count
    1,Mary,1910,F,AK,14
    2,Annie,1910,F,AK,12
    3,Anna,1910,F,AK,10
    4,Margaret,1910,F,AK,8
    5,Helen,1910,F,AK,7
    6,Elsie,1910,F,AK,6
    7,Lucy,1910,F,AK,6
    8,Dorothy,1910,F,AK,5
    9,Mary,1911,F,AK,12


## Baby Names

É bem mais comum fazer uso de DataFrames que já existem em arquivos. Note que o trabalho do cientista de dados nem sempre vai ter tais arquivos prontos. Em várias ocasiões, você vai ter que coletar e organizar os mesmos. Limpeza e coleta de dados é uma parte fundamental do seu trabalho. Durante a matéria, boa parte dos notebooks já vão ter dados prontos.


```python
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/03-Tabelas-e-Tipos-de-Dados/baby.csv', index_col=0)
df
```

    /home/flaviovdf/miniconda3/envs/teaching/lib/python3.6/site-packages/numpy/lib/arraysetops.py:569: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Lucy</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dorothy</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mary</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Margaret</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ruth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Annie</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Elizabeth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Helen</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mary</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Elsie</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Agnes</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anna</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Helen</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Louise</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Jean</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ruth</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Alice</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Esther</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Ethel</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Margaret</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Marie</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mary</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>21</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Elizabeth</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Margaret</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5647397</th>
      <td>Brooks</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647398</th>
      <td>Calvin</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647399</th>
      <td>Cameron</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647400</th>
      <td>Dalton</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647401</th>
      <td>Dawson</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647402</th>
      <td>Edward</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647403</th>
      <td>Elias</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647404</th>
      <td>Gage</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647405</th>
      <td>Hayden</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647406</th>
      <td>Jasper</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647407</th>
      <td>Jose</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647408</th>
      <td>Kaiden</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647409</th>
      <td>Kaleb</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647410</th>
      <td>Kasen</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647411</th>
      <td>Kyson</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647412</th>
      <td>Lukas</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647413</th>
      <td>Myles</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647414</th>
      <td>Nathaniel</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647415</th>
      <td>Nolan</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647416</th>
      <td>Oakley</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647417</th>
      <td>Odin</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647418</th>
      <td>Paxton</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647419</th>
      <td>Raymond</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647420</th>
      <td>Richard</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647421</th>
      <td>Rowan</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647422</th>
      <td>Seth</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647423</th>
      <td>Spencer</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647424</th>
      <td>Tyce</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647425</th>
      <td>Victor</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647426</th>
      <td>Waylon</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5647426 rows × 5 columns</p>
</div>



O método `head` do notebook retorna as primeiras `n` linhas do mesmo. Use tal método para entender seus dados. **Sempre olhe para seus dados.** Note como as linhas abaixo usa o `loc` e `iloc` para entender um pouco a estrutura dos mesmos.


```python
df = pd.read_csv('baby.csv', index_col=0)
df
```

    /home/flaviovdf/miniconda3/envs/teaching/lib/python3.6/site-packages/numpy/lib/arraysetops.py:569: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Lucy</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dorothy</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mary</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Margaret</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ruth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Annie</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Elizabeth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Helen</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mary</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Elsie</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Agnes</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anna</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Helen</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Louise</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Jean</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ruth</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Alice</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Esther</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Ethel</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Margaret</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Marie</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Mary</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>21</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Elizabeth</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Margaret</td>
      <td>1913</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5647397</th>
      <td>Brooks</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647398</th>
      <td>Calvin</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647399</th>
      <td>Cameron</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647400</th>
      <td>Dalton</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647401</th>
      <td>Dawson</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647402</th>
      <td>Edward</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647403</th>
      <td>Elias</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647404</th>
      <td>Gage</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647405</th>
      <td>Hayden</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647406</th>
      <td>Jasper</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647407</th>
      <td>Jose</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647408</th>
      <td>Kaiden</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647409</th>
      <td>Kaleb</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647410</th>
      <td>Kasen</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647411</th>
      <td>Kyson</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647412</th>
      <td>Lukas</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647413</th>
      <td>Myles</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647414</th>
      <td>Nathaniel</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647415</th>
      <td>Nolan</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647416</th>
      <td>Oakley</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647417</th>
      <td>Odin</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647418</th>
      <td>Paxton</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647419</th>
      <td>Raymond</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647420</th>
      <td>Richard</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647421</th>
      <td>Rowan</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647422</th>
      <td>Seth</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647423</th>
      <td>Spencer</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647424</th>
      <td>Tyce</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647425</th>
      <td>Victor</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5647426</th>
      <td>Waylon</td>
      <td>2014</td>
      <td>M</td>
      <td>WY</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5647426 rows × 5 columns</p>
</div>



O método `head` do notebook retorna as primeiras `n` linhas do mesmo. Use tal método para entender seus dados. **Sempre olhe para seus dados.** Note como as linhas abaixo usa o `loc` e `iloc` para entender um pouco a estrutura dos mesmos.


```python
df.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[10:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>Ruth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Annie</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Elizabeth</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Helen</td>
      <td>1911</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mary</td>
      <td>1912</td>
      <td>F</td>
      <td>AK</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0:6]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
      <th>State</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Margaret</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Helen</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elsie</td>
      <td>1910</td>
      <td>F</td>
      <td>AK</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Name', 'Gender']].head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
    </tr>
    <tr>
      <th>Id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Annie</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Anna</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Margaret</td>
      <td>F</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Helen</td>
      <td>F</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Elsie</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>



## Groupby

Vamos responder algumas perguntas com a função groupby. Lembrando a ideia é separar os dados com base em valores comuns, ou seja, agrupar por nomes e realizar alguma operação. O comando abaixo agrupa todos os recem-náscidos por nome. Imagine a mesma fazendo uma operação equivalente ao laço abaixo:

```python
buckets = {}                    # Mapa de dados
names = set(df['Name'])         # Conjunto de nomes únicos
for idx, row in df.iterrows():  # Para cada linha dos dados
    name = row['Name']
    if name not in buckets:
        buckets[name] = []      # Uma lista para cada nome
    buckets[name].append(row)   # Separa a linha para cada nome
```

O código acima é bastante lento!!! O groupby é optimizado. Com base na linha abaixo, o mesmo nem retorna nehum resultado ainda. Apenas um objeto onde podemos fazer agregações.


```python
gb = df.groupby('Name')
type(gb)
```




    pandas.core.groupby.generic.DataFrameGroupBy



Agora posso agregar todos os nomes com alguma operação. Por exemplo, posso somar a quantidade de vezes que cada nome ocorre. Em python, seria o seguinte código.

```python
sum_ = {}                       # Mapa de dados
for name in buckets:            # Para cada nomee
    sum_[name] = 0
    for row in buckets[name]:   # Para cada linha com aquele nome, aggregate (some)
        sum_[name] += row['Count']
```

Observe o resultado da agregação abaixo. Qual o problema com a coluna `Year`??


```python
gb.mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aaban</th>
      <td>2013.500000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Aadan</th>
      <td>2009.750000</td>
      <td>5.750000</td>
    </tr>
    <tr>
      <th>Aadarsh</th>
      <td>2009.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aaden</th>
      <td>2010.015306</td>
      <td>17.479592</td>
    </tr>
    <tr>
      <th>Aadhav</th>
      <td>2014.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Aadhya</th>
      <td>2012.875000</td>
      <td>11.325000</td>
    </tr>
    <tr>
      <th>Aadi</th>
      <td>2008.794872</td>
      <td>8.025641</td>
    </tr>
    <tr>
      <th>Aadil</th>
      <td>2003.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aadin</th>
      <td>2008.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aadit</th>
      <td>2009.666667</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Aaditya</th>
      <td>2008.285714</td>
      <td>6.928571</td>
    </tr>
    <tr>
      <th>Aadya</th>
      <td>2012.638889</td>
      <td>8.333333</td>
    </tr>
    <tr>
      <th>Aadyn</th>
      <td>2010.363636</td>
      <td>6.272727</td>
    </tr>
    <tr>
      <th>Aahan</th>
      <td>2008.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>Aahana</th>
      <td>2011.000000</td>
      <td>8.266667</td>
    </tr>
    <tr>
      <th>Aahil</th>
      <td>2009.750000</td>
      <td>7.812500</td>
    </tr>
    <tr>
      <th>Aahna</th>
      <td>2014.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>Aaiden</th>
      <td>2011.250000</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>Aaima</th>
      <td>2013.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aakash</th>
      <td>2002.250000</td>
      <td>6.568182</td>
    </tr>
    <tr>
      <th>Aalaya</th>
      <td>2014.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aalayah</th>
      <td>2011.411765</td>
      <td>5.823529</td>
    </tr>
    <tr>
      <th>Aaleah</th>
      <td>2012.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aaleyah</th>
      <td>2010.937500</td>
      <td>8.343750</td>
    </tr>
    <tr>
      <th>Aalia</th>
      <td>2012.000000</td>
      <td>5.666667</td>
    </tr>
    <tr>
      <th>Aaliah</th>
      <td>2008.080000</td>
      <td>9.480000</td>
    </tr>
    <tr>
      <th>Aalijah</th>
      <td>2010.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Aaliya</th>
      <td>2008.200000</td>
      <td>6.620000</td>
    </tr>
    <tr>
      <th>Aaliyah</th>
      <td>2004.319876</td>
      <td>71.697723</td>
    </tr>
    <tr>
      <th>Aaliyha</th>
      <td>2006.800000</td>
      <td>5.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Zyire</th>
      <td>2007.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zykeria</th>
      <td>2005.059701</td>
      <td>10.044776</td>
    </tr>
    <tr>
      <th>Zykeriah</th>
      <td>2008.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zykerria</th>
      <td>2003.900000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>Zykia</th>
      <td>2002.538462</td>
      <td>6.923077</td>
    </tr>
    <tr>
      <th>Zykierra</th>
      <td>2007.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zykira</th>
      <td>2006.333333</td>
      <td>6.333333</td>
    </tr>
    <tr>
      <th>Zykiria</th>
      <td>2006.000000</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>Zyla</th>
      <td>2011.705882</td>
      <td>6.235294</td>
    </tr>
    <tr>
      <th>Zylah</th>
      <td>2011.800000</td>
      <td>6.200000</td>
    </tr>
    <tr>
      <th>Zylan</th>
      <td>2011.000000</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>Zylen</th>
      <td>2013.500000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Zyler</th>
      <td>2011.071429</td>
      <td>6.285714</td>
    </tr>
    <tr>
      <th>Zymari</th>
      <td>2009.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zymarion</th>
      <td>2009.666667</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zymere</th>
      <td>2009.785714</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>Zymier</th>
      <td>2010.333333</td>
      <td>5.333333</td>
    </tr>
    <tr>
      <th>Zymiere</th>
      <td>2010.666667</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zymir</th>
      <td>2009.139535</td>
      <td>8.023256</td>
    </tr>
    <tr>
      <th>Zymire</th>
      <td>2012.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>Zyon</th>
      <td>2007.656805</td>
      <td>8.230769</td>
    </tr>
    <tr>
      <th>Zyonna</th>
      <td>2009.818182</td>
      <td>5.545455</td>
    </tr>
    <tr>
      <th>Zyquan</th>
      <td>2005.777778</td>
      <td>5.777778</td>
    </tr>
    <tr>
      <th>Zyquavious</th>
      <td>2010.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Zyra</th>
      <td>2012.142857</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Zyrah</th>
      <td>2012.000000</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>Zyren</th>
      <td>2013.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Zyria</th>
      <td>2006.714286</td>
      <td>5.785714</td>
    </tr>
    <tr>
      <th>Zyriah</th>
      <td>2009.666667</td>
      <td>6.444444</td>
    </tr>
    <tr>
      <th>Zyshonne</th>
      <td>1998.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
<p>30274 rows × 2 columns</p>
</div>



Não faz tanto sentido somar o ano, embora seja um número aqui representa uma categoria. Vamos somar as contagens apenas.


```python
gb.sum()['Count'].sort_values()
```




    Name
    Zyshonne             5
    Makenlee             5
    Makenlie             5
    Makinlee             5
    Makua                5
    Cathaleya            5
    Makyia               5
    Makynzee             5
    Malacai              5
    Catello              5
    Malai                5
    Catcher              5
    Malajah              5
    Maleeha              5
    Maleigh              5
    Castin               5
    Maleko               5
    Malini               5
    Malissia             5
    Malissie             5
    Makaylie             5
    Makay                5
    Makalynn             5
    Makailyn             5
    Mahesh               5
    Caylynn              5
    Caylyn               5
    Mahin                5
    Mahjabeen            5
    Mahreen              5
                    ...   
    Brian          1159034
    Joshua         1174451
    Edward         1212969
    Andrew         1239305
    Kenneth        1261928
    Steven         1272459
    George         1324735
    Mark           1341573
    Paul           1357785
    Anthony        1397105
    Donald         1403439
    Barbara        1424544
    Linda          1448063
    Jennifer       1464686
    Elizabeth      1502094
    Matthew        1551706
    Patricia       1569944
    Daniel         1857096
    Christopher    1997925
    Thomas         2219967
    Charles        2252146
    Joseph         2485220
    Richard        2534949
    David          3562278
    Mary           3740495
    William        3839236
    Michael        4312975
    Robert         4725713
    John           4845414
    James          4957166
    Name: Count, Length: 30274, dtype: int64



E ordenar...


```python
gb.sum()['Count'].sort_values()
```




    Name
    Zyshonne             5
    Makenlee             5
    Makenlie             5
    Makinlee             5
    Makua                5
    Cathaleya            5
    Makyia               5
    Makynzee             5
    Malacai              5
    Catello              5
    Malai                5
    Catcher              5
    Malajah              5
    Maleeha              5
    Maleigh              5
    Castin               5
    Maleko               5
    Malini               5
    Malissia             5
    Malissie             5
    Makaylie             5
    Makay                5
    Makalynn             5
    Makailyn             5
    Mahesh               5
    Caylynn              5
    Caylyn               5
    Mahin                5
    Mahjabeen            5
    Mahreen              5
                    ...   
    Brian          1159034
    Joshua         1174451
    Edward         1212969
    Andrew         1239305
    Kenneth        1261928
    Steven         1272459
    George         1324735
    Mark           1341573
    Paul           1357785
    Anthony        1397105
    Donald         1403439
    Barbara        1424544
    Linda          1448063
    Jennifer       1464686
    Elizabeth      1502094
    Matthew        1551706
    Patricia       1569944
    Daniel         1857096
    Christopher    1997925
    Thomas         2219967
    Charles        2252146
    Joseph         2485220
    Richard        2534949
    David          3562278
    Mary           3740495
    William        3839236
    Michael        4312975
    Robert         4725713
    John           4845414
    James          4957166
    Name: Count, Length: 30274, dtype: int64



É comum, embora mais chato de ler, fazer tudo em uma única chamada. Isto é uma prática que vem do mundo SQL. A chamada abaixo seria o mesmo de:

```sql
SELECT Name, SUM(Count)
FROM baby_table
GROUPBY Name
ORDERBY SUM(Count)
```


```python
df.groupby('Name').sum().sort_values(by='Count')['Count']
```




    Name
    Zyshonne             5
    Makenlee             5
    Makenlie             5
    Makinlee             5
    Makua                5
    Cathaleya            5
    Makyia               5
    Makynzee             5
    Malacai              5
    Catello              5
    Malai                5
    Catcher              5
    Malajah              5
    Maleeha              5
    Maleigh              5
    Castin               5
    Maleko               5
    Malini               5
    Malissia             5
    Malissie             5
    Makaylie             5
    Makay                5
    Makalynn             5
    Makailyn             5
    Mahesh               5
    Caylynn              5
    Caylyn               5
    Mahin                5
    Mahjabeen            5
    Mahreen              5
                    ...   
    Brian          1159034
    Joshua         1174451
    Edward         1212969
    Andrew         1239305
    Kenneth        1261928
    Steven         1272459
    George         1324735
    Mark           1341573
    Paul           1357785
    Anthony        1397105
    Donald         1403439
    Barbara        1424544
    Linda          1448063
    Jennifer       1464686
    Elizabeth      1502094
    Matthew        1551706
    Patricia       1569944
    Daniel         1857096
    Christopher    1997925
    Thomas         2219967
    Charles        2252146
    Joseph         2485220
    Richard        2534949
    David          3562278
    Mary           3740495
    William        3839236
    Michael        4312975
    Robert         4725713
    John           4845414
    James          4957166
    Name: Count, Length: 30274, dtype: int64



Podemos inverter com ::-1


```python
df.groupby(['Name', 'Year']).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Count</th>
    </tr>
    <tr>
      <th>Name</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Aaban</th>
      <th>2013</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Aadan</th>
      <th>2008</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Aadarsh</th>
      <th>2009</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">Aaden</th>
      <th>2005</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>98</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>939</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>1242</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>414</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>228</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>167</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>153</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>180</td>
    </tr>
    <tr>
      <th>Aadhav</th>
      <th>2014</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Aadhya</th>
      <th>2007</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>66</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>135</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>221</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Aadi</th>
      <th>2003</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>8</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>25</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>43</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>33</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>27</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Zyquan</th>
      <th>2008</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Zyquavious</th>
      <th>2010</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Zyra</th>
      <th>2008</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>12</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>13</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Zyrah</th>
      <th>2011</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Zyren</th>
      <th>2013</th>
      <td>6</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">Zyria</th>
      <th>1998</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>11</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>11</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>20</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Zyriah</th>
      <th>2006</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>13</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6</td>
    </tr>
    <tr>
      <th>Zyshonne</th>
      <th>1998</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>548154 rows × 1 columns</p>
</div>



## NBA Salaries e Indexação Booleana

Por fim, vamos explorar alguns dados da NBA para entender a indexação booleana. Vamos carregar os dados da mesma forma que carregamos os dados dos nomes de crianças.


```python
df = pd.read_csv('https://media.githubusercontent.com/media/icd-ufmg/material/master/aulas/03-Tabelas-e-Tipos-de-Dados/nba_salaries.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>POSITION</th>
      <th>TEAM</th>
      <th>SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paul Millsap</td>
      <td>PF</td>
      <td>Atlanta Hawks</td>
      <td>18.671659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Al Horford</td>
      <td>C</td>
      <td>Atlanta Hawks</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tiago Splitter</td>
      <td>C</td>
      <td>Atlanta Hawks</td>
      <td>9.756250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jeff Teague</td>
      <td>PG</td>
      <td>Atlanta Hawks</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kyle Korver</td>
      <td>SG</td>
      <td>Atlanta Hawks</td>
      <td>5.746479</td>
    </tr>
  </tbody>
</table>
</div>



Por fim, vamos indexar nosso DataFrame por booleanos. A linha abaixo pega um vetor de booleanos onde o nome do time é `Houston Rockets`.


```python
df['TEAM'] == 'Houston Rockets'
```




    0      False
    1      False
    2      False
    3      False
    4      False
    5      False
    6      False
    7      False
    8      False
    9      False
    10     False
    11     False
    12     False
    13     False
    14     False
    15     False
    16     False
    17     False
    18     False
    19     False
    20     False
    21     False
    22     False
    23     False
    24     False
    25     False
    26     False
    27     False
    28     False
    29     False
           ...  
    387    False
    388    False
    389    False
    390    False
    391    False
    392    False
    393    False
    394    False
    395    False
    396    False
    397    False
    398    False
    399    False
    400    False
    401    False
    402    False
    403    False
    404    False
    405    False
    406    False
    407    False
    408    False
    409    False
    410    False
    411    False
    412    False
    413    False
    414    False
    415    False
    416    False
    Name: TEAM, Length: 417, dtype: bool



Podemos usar tal vetor para filtrar nosso DataFrame. A linha abaixo é o mesmo de um:

```sql
SELECT *
FROM table
WHERE TEAM = 'Houston Rockets'
```


```python
filtro = df['TEAM'] == 'Houston Rockets'
df[filtro]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>POSITION</th>
      <th>TEAM</th>
      <th>SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>Dwight Howard</td>
      <td>C</td>
      <td>Houston Rockets</td>
      <td>22.359364</td>
    </tr>
    <tr>
      <th>132</th>
      <td>James Harden</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>15.756438</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Ty Lawson</td>
      <td>PG</td>
      <td>Houston Rockets</td>
      <td>12.404495</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Corey Brewer</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>8.229375</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Trevor Ariza</td>
      <td>SF</td>
      <td>Houston Rockets</td>
      <td>8.193030</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Patrick Beverley</td>
      <td>PG</td>
      <td>Houston Rockets</td>
      <td>6.486486</td>
    </tr>
    <tr>
      <th>137</th>
      <td>K.J. McDaniels</td>
      <td>SG</td>
      <td>Houston Rockets</td>
      <td>3.189794</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Terrence Jones</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>2.489530</td>
    </tr>
    <tr>
      <th>139</th>
      <td>Donatas Motiejunas</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>2.288205</td>
    </tr>
    <tr>
      <th>140</th>
      <td>Sam Dekker</td>
      <td>SF</td>
      <td>Houston Rockets</td>
      <td>1.646400</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Clint Capela</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>1.242720</td>
    </tr>
    <tr>
      <th>142</th>
      <td>Montrezl Harrell</td>
      <td>PF</td>
      <td>Houston Rockets</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Assim como pegar os salários maior do que um certo valor!


```python
df[df['SALARY'] > 20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PLAYER</th>
      <th>POSITION</th>
      <th>TEAM</th>
      <th>SALARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>Joe Johnson</td>
      <td>SF</td>
      <td>Brooklyn Nets</td>
      <td>24.894863</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Derrick Rose</td>
      <td>PG</td>
      <td>Chicago Bulls</td>
      <td>20.093064</td>
    </tr>
    <tr>
      <th>72</th>
      <td>LeBron James</td>
      <td>SF</td>
      <td>Cleveland Cavaliers</td>
      <td>22.970500</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Dwight Howard</td>
      <td>C</td>
      <td>Houston Rockets</td>
      <td>22.359364</td>
    </tr>
    <tr>
      <th>156</th>
      <td>Chris Paul</td>
      <td>PG</td>
      <td>Los Angeles Clippers</td>
      <td>21.468695</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Kobe Bryant</td>
      <td>SF</td>
      <td>Los Angeles Lakers</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>201</th>
      <td>Chris Bosh</td>
      <td>PF</td>
      <td>Miami Heat</td>
      <td>22.192730</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Carmelo Anthony</td>
      <td>SF</td>
      <td>New York Knicks</td>
      <td>22.875000</td>
    </tr>
    <tr>
      <th>268</th>
      <td>Kevin Durant</td>
      <td>SF</td>
      <td>Oklahoma City Thunder</td>
      <td>20.158622</td>
    </tr>
  </tbody>
</table>
</div>



## Exercícios

Abaixo temos algumas chamadas em pandas. Tente explicar cada uma delas.


```python
df[['POSITION', 'SALARY']].groupby('POSITION').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SALARY</th>
    </tr>
    <tr>
      <th>POSITION</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C</th>
      <td>6.082913</td>
    </tr>
    <tr>
      <th>PF</th>
      <td>4.951344</td>
    </tr>
    <tr>
      <th>PG</th>
      <td>5.165487</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>5.532675</td>
    </tr>
    <tr>
      <th>SG</th>
      <td>3.988195</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['TEAM', 'SALARY']].groupby('TEAM').mean().sort_values('SALARY')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SALARY</th>
    </tr>
    <tr>
      <th>TEAM</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Phoenix Suns</th>
      <td>2.971813</td>
    </tr>
    <tr>
      <th>Utah Jazz</th>
      <td>3.095993</td>
    </tr>
    <tr>
      <th>Portland Trail Blazers</th>
      <td>3.246206</td>
    </tr>
    <tr>
      <th>Philadelphia 76ers</th>
      <td>3.267796</td>
    </tr>
    <tr>
      <th>Boston Celtics</th>
      <td>3.352367</td>
    </tr>
    <tr>
      <th>Milwaukee Bucks</th>
      <td>4.019873</td>
    </tr>
    <tr>
      <th>Detroit Pistons</th>
      <td>4.221176</td>
    </tr>
    <tr>
      <th>Toronto Raptors</th>
      <td>4.392507</td>
    </tr>
    <tr>
      <th>Brooklyn Nets</th>
      <td>4.408229</td>
    </tr>
    <tr>
      <th>Denver Nuggets</th>
      <td>4.459243</td>
    </tr>
    <tr>
      <th>Memphis Grizzlies</th>
      <td>4.466497</td>
    </tr>
    <tr>
      <th>Charlotte Hornets</th>
      <td>4.672355</td>
    </tr>
    <tr>
      <th>Indiana Pacers</th>
      <td>4.822694</td>
    </tr>
    <tr>
      <th>Atlanta Hawks</th>
      <td>4.969507</td>
    </tr>
    <tr>
      <th>New Orleans Pelicans</th>
      <td>5.032163</td>
    </tr>
    <tr>
      <th>Minnesota Timberwolves</th>
      <td>5.065186</td>
    </tr>
    <tr>
      <th>Los Angeles Clippers</th>
      <td>5.082624</td>
    </tr>
    <tr>
      <th>Washington Wizards</th>
      <td>5.296912</td>
    </tr>
    <tr>
      <th>New York Knicks</th>
      <td>5.338846</td>
    </tr>
    <tr>
      <th>Orlando Magic</th>
      <td>5.544567</td>
    </tr>
    <tr>
      <th>Dallas Mavericks</th>
      <td>5.978414</td>
    </tr>
    <tr>
      <th>Oklahoma City Thunder</th>
      <td>6.052010</td>
    </tr>
    <tr>
      <th>Sacramento Kings</th>
      <td>6.216808</td>
    </tr>
    <tr>
      <th>Los Angeles Lakers</th>
      <td>6.237086</td>
    </tr>
    <tr>
      <th>San Antonio Spurs</th>
      <td>6.511698</td>
    </tr>
    <tr>
      <th>Chicago Bulls</th>
      <td>6.568407</td>
    </tr>
    <tr>
      <th>Golden State Warriors</th>
      <td>6.720367</td>
    </tr>
    <tr>
      <th>Miami Heat</th>
      <td>6.794056</td>
    </tr>
    <tr>
      <th>Houston Rockets</th>
      <td>7.107153</td>
    </tr>
    <tr>
      <th>Cleveland Cavaliers</th>
      <td>10.231241</td>
    </tr>
  </tbody>
</table>
</div>



## Merge

Agora, vamos explorar algumas chamadas que fazem opereações de merge.


```python
people = pd.DataFrame(
    [["Joey",      "blue",       42,  "M"],
     ["Weiwei",    "blue",       50,  "F"],
     ["Joey",      "green",       8,  "M"],
     ["Karina",    "green",  np.nan,  "F"],
     ["Fernando",  "pink",        9,  "M"],
     ["Nhi",       "blue",        3,  "F"],
     ["Sam",       "pink",   np.nan,  "M"]], 
    columns = ["Name", "Color", "Age", "Gender"])
people
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Color</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joey</td>
      <td>blue</td>
      <td>42.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Joey</td>
      <td>green</td>
      <td>8.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Karina</td>
      <td>green</td>
      <td>NaN</td>
      <td>F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fernando</td>
      <td>pink</td>
      <td>9.0</td>
      <td>M</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nhi</td>
      <td>blue</td>
      <td>3.0</td>
      <td>F</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sam</td>
      <td>pink</td>
      <td>NaN</td>
      <td>M</td>
    </tr>
  </tbody>
</table>
</div>




```python
email = pd.DataFrame(
    [["Deb",  "deborah_nolan@berkeley.edu"],
     ["Sam",  np.nan],
     ["John", "doe@nope.com"],
     ["Joey", "jegonzal@cs.berkeley.edu"],
     ["Weiwei", "weiwzhang@berkeley.edu"],
     ["Weiwei", np.nan],
     ["Karina", "kgoot@berkeley.edu"]], 
    columns = ["User Name", "Email"])
email
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Deb</td>
      <td>deborah_nolan@berkeley.edu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sam</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>John</td>
      <td>doe@nope.com</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Weiwei</td>
      <td>weiwzhang@berkeley.edu</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Weiwei</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Karina</td>
      <td>kgoot@berkeley.edu</td>
    </tr>
  </tbody>
</table>
</div>




```python
people.merge(email, 
             how = "inner",
             left_on = "Name", right_on = "User Name")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Color</th>
      <th>Age</th>
      <th>Gender</th>
      <th>User Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joey</td>
      <td>blue</td>
      <td>42.0</td>
      <td>M</td>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joey</td>
      <td>green</td>
      <td>8.0</td>
      <td>M</td>
      <td>Joey</td>
      <td>jegonzal@cs.berkeley.edu</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
      <td>Weiwei</td>
      <td>weiwzhang@berkeley.edu</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Weiwei</td>
      <td>blue</td>
      <td>50.0</td>
      <td>F</td>
      <td>Weiwei</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Karina</td>
      <td>green</td>
      <td>NaN</td>
      <td>F</td>
      <td>Karina</td>
      <td>kgoot@berkeley.edu</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sam</td>
      <td>pink</td>
      <td>NaN</td>
      <td>M</td>
      <td>Sam</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Para exploração futura

* Veja a documentação do pandas. https://pandas.pydata.org/
* O livro do Jake Vanderplas explora várias funções pandas. https://jakevdp.github.io/PythonDataScienceHandbook/
