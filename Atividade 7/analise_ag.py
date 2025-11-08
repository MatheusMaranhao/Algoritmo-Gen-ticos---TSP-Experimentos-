import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import time
import os # Para criar uma pasta para os gráficos

# -----------------------------------------------------------------
# DADOS DO PROBLEMA (A instância USA13)
# -----------------------------------------------------------------
USA13 = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]
N_CIDADES = 13

# -----------------------------------------------------------------
# FUNÇÃO DE FITNESS (MINIMIZAR A DISTÂNCIA)
# -----------------------------------------------------------------
def calculate_total_distance(route):
    """Calcula a distância total de uma rota (nosso fitness)."""
    total_distance = 0
    for i in range(N_CIDADES - 1):
        total_distance += USA13[route[i]][route[i+1]]
    total_distance += USA13[route[-1]][route[0]] # Volta para o início
    return total_distance

# -----------------------------------------------------------------
# FUNÇÕES DO AG PARA TSP (base da Atividade 6)
# -----------------------------------------------------------------

def criar_individuo():
    """Cria uma rota aleatória (permutação) das cidades."""
    individuo = list(range(N_CIDADES))
    random.shuffle(individuo)
    return individuo

def selecao_torneio(populacao_com_fitness, k_torneio):
    """Sorteia K competidores e retorna o de MENOR fitness."""
    participantes = random.sample(populacao_com_fitness, k_torneio)
    participantes.sort(key=lambda item: item[1]) # Ordena pelo fitness (distância)
    return participantes[0][0] # Retorna o melhor (menor distância)

def crossover_ox(pai1, pai2, taxa_crossover):
    """Implementa o Ordered Crossover (OX) com uma taxa."""
    if random.random() > taxa_crossover:
        return pai1.copy(), pai2.copy()

    tamanho = len(pai1)
    filho1, filho2 = [None] * tamanho, [None] * tamanho
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    
    filho1[inicio:fim+1] = pai1[inicio:fim+1]
    filho2[inicio:fim+1] = pai2[inicio:fim+1]
    
    # Preenche o Filho 1 com genes do Pai 2
    idx_pai2 = 0
    idx_filho1 = 0
    while None in filho1:
        if filho1[idx_filho1] is not None:
            idx_filho1 = (idx_filho1 + 1) % tamanho
            continue
        gene_pai2 = pai2[idx_pai2]
        idx_pai2 = (idx_pai2 + 1) % tamanho
        if gene_pai2 not in filho1:
            filho1[idx_filho1] = gene_pai2

    # Preenche o Filho 2 com genes do Pai 1
    idx_pai1 = 0
    idx_filho2 = 0
    while None in filho2:
        if filho2[idx_filho2] is not None:
            idx_filho2 = (idx_filho2 + 1) % tamanho
            continue
        gene_pai1 = pai1[idx_pai1]
        idx_pai1 = (idx_pai1 + 1) % tamanho
        if gene_pai1 not in filho2:
            filho2[idx_filho2] = gene_pai1
            
    return filho1, filho2

def mutacao_swap(individuo, taxa_mutacao):
    """Aplica a mutação de troca (swap) com base na taxa."""
    individuo_mutado = individuo.copy()
    # A taxa é a chance do indivíduo sofrer UMA troca
    if random.random() < taxa_mutacao:
        idx1, idx2 = random.sample(range(N_CIDADES), 2)
        # Jeito pythônico de fazer a troca
        individuo_mutado[idx1], individuo_mutado[idx2] = individuo_mutado[idx2], individuo_mutado[idx1]
        
    return individuo_mutado

# -----------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO AG (Versão 7, mais flexível)
# -----------------------------------------------------------------

def executar_ag_tsp(pop_size, n_geracoes, taxa_cross, taxa_mut, k_torneio, taxa_elitismo):
    
    # 1. Inicializa a população
    populacao = [criar_individuo() for _ in range(pop_size)]
    
    historico_melhor_fitness = []
    historico_diversidade = [] # Para o Experimento 3
    
    # Calcula o NÚMERO de indivíduos de elite
    n_elite = int(pop_size * taxa_elitismo)

    for geracao in range(n_geracoes):
        # 2. Avalia a população
        pop_com_fitness = [(ind, calculate_total_distance(ind)) for ind in populacao]
        pop_com_fitness.sort(key=lambda item: item[1]) # Ordena do melhor para o pior
        
        # 3. Guarda os dados da geração
        historico_melhor_fitness.append(pop_com_fitness[0][1]) # O fitness do melhor
        
        # Cálculo da diversidade (Exp 3)
        individuos_unicos = set(tuple(ind) for ind in populacao)
        historico_diversidade.append(len(individuos_unicos))
        
        nova_populacao = []
        
        # 4. Elitismo (se houver)
        if n_elite > 0:
            for i in range(n_elite):
                nova_populacao.append(pop_com_fitness[i][0]) # Pega a rota
        
        # 5. Gera o restante da população
        while len(nova_populacao) < pop_size:
            pai1 = selecao_torneio(pop_com_fitness, k_torneio)
            pai2 = selecao_torneio(pop_com_fitness, k_torneio)
            filho1, filho2 = crossover_ox(pai1, pai2, taxa_cross)
            filho1 = mutacao_swap(filho1, taxa_mut)
            filho2 = mutacao_swap(filho2, taxa_mut)
            
            nova_populacao.append(filho1)
            if len(nova_populacao) < pop_size:
                nova_populacao.append(filho2)
        
        populacao = nova_populacao
        
    fitness_final = pop_com_fitness[0][1] # Pega o melhor da última geração
    
    return fitness_final, historico_melhor_fitness, historico_diversidade

# -----------------------------------------------------------------
# FUNÇÕES DE PLOTAGEM (para o relatório)
# -----------------------------------------------------------------

def plotar_convergencia(resultados_experimento, titulo, pasta_graficos):
    """Plota o gráfico de linha de convergência."""
    plt.figure(figsize=(10, 6))
    for nome_config, dados in resultados_experimento.items():
        media_convergencia = np.mean(np.array(dados["convergencia"]), axis=0)
        plt.plot(media_convergencia, label=nome_config)
        
    plt.title(f"Velocidade de Convergência - {titulo}")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness (Distância Média)")
    plt.legend()
    plt.grid(True, linestyle='--')
    caminho = os.path.join(pasta_graficos, f"{titulo}_convergencia.png")
    plt.savefig(caminho)
    print(f"Gráfico de convergência salvo: {caminho}")
    plt.close()

def plotar_boxplot(resultados_experimento, titulo, pasta_graficos):
    """Plota o boxplot dos resultados finais."""
    data_boxplot = []
    for nome_config, dados in resultados_experimento.items():
        for fitness_final in dados["finais"]:
            data_boxplot.append({"Configuração": nome_config, "Fitness Final": fitness_final})
            
    df = pd.DataFrame(data_boxplot)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Configuração", y="Fitness Final")
    sns.stripplot(data=df, x="Configuração", y="Fitness Final", color=".3", alpha=0.5)
    plt.title(f"Qualidade da Solução Final - {titulo}")
    caminho = os.path.join(pasta_graficos, f"{titulo}_boxplot.png")
    plt.savefig(caminho)
    print(f"Gráfico de boxplot salvo: {caminho}")
    plt.close()

def plotar_diversidade(resultados_experimento, titulo, pasta_graficos):
    """Plota o gráfico de linha da diversidade (Especial para Exp 3)."""
    plt.figure(figsize=(10, 6))
    for nome_config, dados in resultados_experimento.items():
        media_diversidade = np.mean(np.array(dados["diversidade"]), axis=0)
        plt.plot(media_diversidade, label=nome_config)
        
    plt.title(f"Diversidade da População - {titulo}")
    plt.xlabel("Geração")
    plt.ylabel("Indivíduos Únicos (Média)")
    plt.legend()
    plt.grid(True, linestyle='--')
    caminho = os.path.join(pasta_graficos, f"{titulo}_diversidade.png")
    plt.savefig(caminho)
    print(f"Gráfico de diversidade salvo: {caminho}")
    plt.close()

# -----------------------------------------------------------------
# EXECUTOR DOS EXPERIMENTOS (O "Painel de Controle")
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    # --- Configurações Gerais ---
    N_EXECUCOES = 30 # Conforme pedido
    # *** AQUI A MUDANÇA ***
    N_GERACOES_FIXO = 400 # Fixo, conforme Atividade 6
    
    # --- Parâmetros Base (da Atividade 6) ---
    POP_SIZE_BASE = 50
    TAXA_CROSS_BASE = 0.90 
    TAXA_MUT_BASE = 0.05   # 5%
    K_TORNEIO_BASE = 3
    # *** AQUI A MUDANÇA ***
    TAXA_ELIT_BASE = 0.10  # 10% (pois 5 indivíduos de 50 = 10%)

    # Cria uma pasta para salvar os gráficos
    PASTA_GRAFICOS = "graficos_atividade_7"
    if not os.path.exists(PASTA_GRAFICOS):
        os.makedirs(PASTA_GRAFICOS)
    
    print("Iniciando Atividade 7: Análise de Parâmetros do AG para TSP...")
    print(f"Cada um dos 4 experimentos será executado {N_EXECUCOES} vezes.")
    print(f"Usando {N_GERACOES_FIXO} gerações fixas (conforme Atv. 6).")
    print(f"Gráficos serão salvos em '{PASTA_GRAFICOS}/'")
    print("Isso vai levar um bom tempo...")
    
    start_total = time.time()
    
    # --- Experimento 1: Tamanho da População ---
    print("\n--- Iniciando Experimento 1: Tamanho da População ---")
    params_pop = [20, 50, 100]
    resultados_exp1 = {}
    
    for pop_size in params_pop:
        nome_config = f"Pop={pop_size}"
        print(f"Testando {nome_config}...")
        tempos = []
        finais = []
        convergencias = []
        
        for i in range(N_EXECUCOES):
            start_run = time.time()
            # O Elitismo é 10% *da população atual*
            fitness, hist_fit, _ = executar_ag_tsp(
                pop_size=pop_size, # Variável
                n_geracoes=N_GERACOES_FIXO, 
                taxa_cross=TAXA_CROSS_BASE, 
                taxa_mut=TAXA_MUT_BASE, 
                k_torneio=K_TORNEIO_BASE, 
                taxa_elitismo=TAXA_ELIT_BASE # Fixo em 10%
            )
            end_run = time.time()
            tempos.append(end_run - start_run)
            finais.append(fitness)
            convergencias.append(hist_fit)
        
        resultados_exp1[nome_config] = {
            "finais": finais,
            "convergencia": convergencias,
            "tempos_execucao": tempos # Guardar o tempo
        }

    # Análise Exp 1
    print("\nAnálise de Tempo (Exp 1):")
    for nome_config, dados in resultados_exp1.items():
        print(f"  {nome_config}: Tempo médio: {np.mean(dados['tempos_execucao']):.3f}s")
    
    plotar_convergencia(resultados_exp1, "Exp1_Tamanho_Populacao", PASTA_GRAFICOS)
    plotar_boxplot(resultados_exp1, "Exp1_Tamanho_Populacao", PASTA_GRAFICOS)

    # --- Experimento 2: Taxa de Mutação ---
    print("\n--- Iniciando Experimento 2: Taxa de Mutação ---")
    params_mutacao = [0.01, 0.05, 0.10, 0.20] # 1%, 5%, 10%, 20%
    resultados_exp2 = {}

    for taxa_mut in params_mutacao:
        nome_config = f"Mutacao={taxa_mut*100:.0f}%"
        print(f"Testando {nome_config}...")
        finais = []
        convergencias = []
        
        for i in range(N_EXECUCOES):
            fitness, hist_fit, _ = executar_ag_tsp(
                pop_size=POP_SIZE_BASE, 
                n_geracoes=N_GERACOES_FIXO, 
                taxa_cross=TAXA_CROSS_BASE,
                taxa_mut=taxa_mut, # Variável 
                k_torneio=K_TORNEIO_BASE, 
                taxa_elitismo=TAXA_ELIT_BASE
            )
            finais.append(fitness)
            convergencias.append(hist_fit)

        resultados_exp2[nome_config] = {
            "finais": finais,
            "convergencia": convergencias
        }
    
    plotar_convergencia(resultados_exp2, "Exp2_Taxa_Mutacao", PASTA_GRAFICOS)
    plotar_boxplot(resultados_exp2, "Exp2_Taxa_Mutacao", PASTA_GRAFICOS)

    # --- Experimento 3: Tamanho do Torneio ---
    print("\n--- Iniciando Experimento 3: Tamanho do Torneio ---")
    params_torneio = [2, 3, 5, 7]
    resultados_exp3 = {}

    for k_torneio in params_torneio:
        nome_config = f"Torneio={k_torneio}"
        print(f"Testando {nome_config}...")
        finais = []
        convergencias = []
        diversidades = []

        for i in range(N_EXECUCOES):
            fitness, hist_fit, hist_div = executar_ag_tsp(
                pop_size=POP_SIZE_BASE, 
                n_geracoes=N_GERACOES_FIXO, 
                taxa_cross=TAXA_CROSS_BASE,
                taxa_mut=TAXA_MUT_BASE, 
                k_torneio=k_torneio, # Variável
                taxa_elitismo=TAXA_ELIT_BASE
            )
            finais.append(fitness)
            convergencias.append(hist_fit)
            diversidades.append(hist_div) # Guarda o histórico de diversidade

        resultados_exp3[nome_config] = {
            "finais": finais,
            "convergencia": convergencias,
            "diversidade": diversidades # Salva aqui
        }

    plotar_convergencia(resultados_exp3, "Exp3_Tamanho_Torneio", PASTA_GRAFICOS)
    plotar_boxplot(resultados_exp3, "Exp3_Tamanho_Torneio", PASTA_GRAFICOS)
    plotar_diversidade(resultados_exp3, "Exp3_Tamanho_Torneio", PASTA_GRAFICOS) # Gráfico especial

    # --- Experimento 4: Taxa de Elitismo ---
    print("\n--- Iniciando Experimento 4: Elitismo ---")
    params_elitismo = [0.0, 0.01, 0.05, 0.10] # 0%, 1%, 5%, 10%
    resultados_exp4 = {}

    for taxa_elit in params_elitismo:
        # Nota: O teste '10%' é o mesmo da configuração base
        nome_config = f"Elitismo={taxa_elit*100:.0f}%"
        print(f"Testando {nome_config}...")
        finais = []
        convergencias = []

        for i in range(N_EXECUCOES):
            fitness, hist_fit, _ = executar_ag_tsp(
                pop_size=POP_SIZE_BASE, 
                n_geracoes=N_GERACOES_FIXO, 
                taxa_cross=TAXA_CROSS_BASE,
                taxa_mut=TAXA_MUT_BASE, 
                k_torneio=K_TORNEIO_BASE, 
                taxa_elitismo=taxa_elit # Variável
            )
            finais.append(fitness)
            convergencias.append(hist_fit)

        resultados_exp4[nome_config] = {
            "finais": finais,
            "convergencia": convergencias
        }
    
    plotar_convergencia(resultados_exp4, "Exp4_Elitismo", PASTA_GRAFICOS)
    plotar_boxplot(resultados_exp4, "Exp4_Elitismo", PASTA_GRAFICOS)

    end_total = time.time()
    print(f"\n--- TODOS OS EXPERIMENTOS CONCLUÍDOS em {end_total - start_total:.2f} segundos ---")