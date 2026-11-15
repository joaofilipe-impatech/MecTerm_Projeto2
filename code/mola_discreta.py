"""
Módulo contendo todas as funções utilizadas durante o desenvolvimento do Projeto 1 de Mecânica e Termodinâmica.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

np.set_printoptions(precision=3, suppress=True)


def construir_matriz(N, k_n=None, k=1, m=1, L=1):
    """
    Constrói a matriz A de um cenário do problema com os parâmetros dados
    """
    if k_n is None:
        k_n = np.zeros(N)

    cenario = dict()
    cenario["A"] = np.diag(2. + k_n / k)
    cenario["A"] += np.diag(-np.ones(N - 1), 1)
    cenario["A"] += np.diag(-np.ones(N - 1), -1)
    cenario["A"][0, 0] -= 1
    cenario["A"][-1, -1] -= 1

    cenario["k"] = k
    cenario["m"] = m
    cenario["L"] = L

    return cenario


def inserir_autovalores(cenario):
    """
    Modifica o dicionário de um cenário, inserindo os autovalores e autovetores correspondentes
    """
    cenario["lambda_n"], cenario["u_n"] = np.linalg.eigh(cenario["A"])
    cenario["lambda_n"] = np.maximum(cenario["lambda_n"], 0)


def configurar_estado_inicial(cenario, x_0=None, v_0=None):
    """
    Modifica o dicionário de um cenário, inserindo as posições e velocidades iniciais e os coeficientes correspondentes
    """
    if v_0 is None:
        v_0 = np.zeros(len(cenario["A"]))
    if x_0 is None:
        x_0 = np.zeros(len(cenario["A"]))
    elif type(x_0) is int:
        x_0 = cenario["u_n"][:, x_0]
    cenario["x_0"], cenario["v_0"] = x_0, v_0

    a_0 = cenario["u_n"].T @ x_0
    b_0 = cenario["u_n"].T @ v_0

    cenario["a_0"], cenario["b_0"] = a_0, b_0


def inserir_omega_n(cenario):
    cenario["omega_n"] = np.sqrt(cenario["k"]/cenario["m"] * cenario["lambda_n"])


def criar_cenario(N, k_n=None, k=1, x_0=None, v_0=None, m=1, L=1):
    """
    Cria um novo cenário por completo
    """
    cenario = construir_matriz(N, k_n, k, m, L)
    inserir_autovalores(cenario)
    inserir_omega_n(cenario)
    configurar_estado_inicial(cenario, x_0, v_0)

    return cenario


def x_t(cenario, tempo):
    a_0 = cenario["a_0"][:, None]
    b_0 = cenario["b_0"][:, None]
    omega_n = cenario["omega_n"][:, None]
    u_n = cenario["u_n"]

    tol = 1e-10
    zero_mask = (omega_n < tol)
    
    omega_safe = np.where(zero_mask, 1.0, omega_n)
    
    q_osc = a_0 * np.cos(omega_safe * tempo) + (b_0 / omega_safe) * np.sin(omega_safe * tempo)
    q_trans = a_0 + b_0 * tempo
    
    q = np.where(zero_mask, q_trans, q_osc)
    
    return u_n @ q
    

def v_t(cenario, tempo):
    a_0 = cenario["a_0"][:, None]
    b_0 = cenario["b_0"][:, None]
    omega_n = cenario["omega_n"][:, None]
    u_n = cenario["u_n"]

    tol = 1e-10
    zero_mask = (omega_n < tol)

    omega_safe = np.where(zero_mask, 1.0, omega_n)

    q_osc = -omega_n * a_0 * np.sin(omega_safe * tempo) + b_0 * np.cos(omega_safe * tempo)
    q_trans = a_0 + b_0
    
    q = np.where(zero_mask, q_trans, q_osc)
    
    return u_n @ q


def descrever_movimento(cenario, instantes):
    x_instantes = x_t(cenario, instantes)
    v_instantes = v_t(cenario, instantes)
    return x_instantes, v_instantes


def plot_espaco_tempo(cenario, tempo):
    x, _ = descrever_movimento(cenario, tempo)

    for i, posicao in enumerate(x):
        plt.plot(posicao + (i + 1) * cenario["L"], tempo, label=f"Massa {i+1}")
    for i, posicao in enumerate(x):
        plt.vlines((i+1) * cenario["L"], tempo[0], tempo[-1], colors="black", linestyles="--", alpha=0.3)
    # plt.legend()
    plt.xlabel("Posição das massas")
    plt.ylabel("Tempo")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Diagrama Espaço-Tempo (Linhas de Universo das Massas)')
    plt.show()
    plt.close()


def obter_ep_ideal(cenario):
    A, k, x_0 = cenario["A"], cenario["k"], cenario["x_0"]
    return 0.5 * k * np.dot(x_0, A @ x_0)


def obter_ep(cenario, tempo):
    A, k, x = cenario["A"], cenario["k"], x_t(cenario, tempo)
    return 0.5 * k * np.sum(x * (A @ x), axis=0) 


def obter_ek_ideal(cenario):
    m, v_0 = cenario["m"], cenario["v_0"]
    return 0.5 * m * np.sum(v_0**2)


def obter_ek(cenario, tempo):
    m, v = cenario["m"], v_t(cenario, tempo)
    return 0.5 * m * np.sum(v * v, axis=0)


def obter_em(cenario, tempo):
    return obter_ep(cenario, tempo) + obter_ek(cenario, tempo)


def obter_em_ideal(cenario):
    return obter_ep_ideal(cenario) + obter_ek_ideal(cenario)


def animar_massas(x_matrix, cenario):
    L = cenario["L"]
    N, T = x_matrix.shape
    pos_eq = np.arange(1, N + 1) * L
    
    fig, ax = plt.subplots(figsize=(12, 2))
    
    max_deslocamento = np.max(np.abs(x_matrix))
    ax.set_xlim(pos_eq[0] - max_deslocamento - 0.5, pos_eq[-1] + max_deslocamento + 0.5)
    
    ax.set_ylim(-1, 1)
    ax.set_yticks([]) 
    ax.set_xlabel("Posição Real ($x$)")
    ax.set_title("Movimento das Massas (1D)")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    pontos, = ax.plot([], [], 'o', color='royalblue', markersize=8)
    
    def init():
        pontos.set_data([], [])
        return (pontos,)
    
    def animate(frame):
        pos_real = pos_eq + x_matrix[:, frame]
        pontos.set_data(pos_real, np.zeros(N))
        return (pontos,)
    
    passo_frame = max(1, T // 200) 
    
    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=np.arange(0, T, passo_frame), 
                        interval=50, blit=True)
    
    plt.close(fig) 

    return HTML(ani.to_jshtml())


def plot_mapa_cores(x_matrix, tempos):
    N, T = x_matrix.shape
    
    # Cria um array com os índices das massas (1 até N)
    massas = np.arange(1, N + 1)
    
    plt.figure(figsize=(10, 6))
    
    # pcolormesh desenha a grade de cores. 
    # Passamos x_matrix.T (transposta) porque a função espera o formato (tempo, massas) 
    # para bater com os eixos Y e X, respectivamente.
    # O colormap 'RdBu_r' (Red-Blue invertido) é divergente: centra o branco no zero.
    
    # Para garantir que o branco seja exatamente zero, definimos vmin e vmax simétricos
    limite = np.max(np.abs(x_matrix))
    mapa = plt.pcolormesh(massas, tempos, x_matrix.T, 
                          cmap='RdBu_r', vmin=-limite, vmax=limite, shading='auto')
    
    # Adiciona a barra lateral para interpretar as cores
    cbar = plt.colorbar(mapa)
    cbar.set_label('Deslocamento $x_i(t)$')
    
    plt.xlabel('Índice da Massa ($i$)')
    plt.ylabel('Tempo ($t$)')
    plt.title('Mapa de Cores dos Deslocamentos')
    
    # Opcional: inverte o eixo Y para o tempo crescer de cima para baixo
    # plt.gca().invert_yaxis()
    
    plt.show()


def plot_energia(cenario, tempo):
    ep = obter_ep(cenario, tempo)
    ek = obter_ek(cenario, tempo)

    plt.plot(ep, tempo, label=f"Energia Potencial")
    plt.plot(ek, tempo, label=f"Energia Cinética")
    plt.plot(ek + ep, tempo, label=f"Energia Mecânica")
    plt.legend()
    plt.xlabel("Energia do sistema")
    plt.ylabel("Tempo")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Gráfico de Energia')
    plt.show()
    plt.close()


def plot_energia_modos(cenario):
    a_0 = cenario["a_0"]
    b_0 = cenario["b_0"]
    omega_n = cenario["omega_n"]
    m = cenario["m"]
    
    energia_modos = 0.5 * m * (b_0**2) + 0.5 * m * (omega_n**2) * (a_0**2)
    
    modos = np.arange(1, len(omega_n) + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.bar(modos, energia_modos, color='darkorange', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Índice do Modo Normal ($n$)')
    plt.ylabel('Energia do Modo ($E_n$)')
    plt.title('Espectro de Energia (Energia contida em cada modo)')
    
    plt.xticks(modos)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()