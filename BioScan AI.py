import os
import sys
import re
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- CONFIGURAÇÃO DE AMBIENTE ---
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")

# --- BANCO DE DADOS INICIAL ---
banco_receptores = {
    "Receptor ACE2": "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQ",
    "Receptor CD4": "MNRGVPFRHLLLVLQLALLPAATQGKKVVLGKKGDTVELTCTASQKKSIQFHWKNSNQIKILGNQGSFLTKGPSKLNDRADSRRSLWDQGNFPLIIKNLKIED",
    "Receptor NTCP": "MEAHNASAPNTSDPAVGVAVVIMLMLGLLVLAIFGWNMKLLKTVLKVLPTLFLGLLVGALVLAIFGWNMKLLKTVLKVLPTLFLGLLVGALVLAIFGWNMKL",
    "Receptor Sialic": "MKNLLYMAALVLLALVAVADRDPGKVFGLVLLGGVILLVLAIFGWNMKLLKTVLKVLPTLFLGLLVGALVLAIFGWNMKLLKTVLKVLPTLFLGLLVGAL",
    "Receptor CCR5": "MDYQVSSPIYDINYYTSEPCQKINVKQIAARLLPPLYSLVFIFGFVGNMLVILILINCKRLKSMTDIYLLNLAISDLFFLLTVPFWAHYAAAQWDFGNTMCQ"
}

def limpar_sequencia(texto):
    linhas = texto.splitlines()
    sequencia_limpa = "".join([l for l in linhas if not l.startswith(">")])
    return re.sub(r'[^A-Z]', '', sequencia_limpa.upper())

class BioScanApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # --- CONFIGURAÇÃO DE TÍTULO E JANELA ---
        self.title("BioScan AI- Predição Viral")
        self.geometry("800x500")
        self.minsize(800, 500) 
        self.resizable(True, True) 
        ctk.set_appearance_mode("light") 

        # --- LOCALIZADOR DE CAMINHO (PARA ÍCONE E LOGO) ---
        if hasattr(sys, '_MEIPASS'):
            caminho_base = sys._MEIPASS
        else:
            try:
                caminho_base = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                caminho_base = os.path.abspath(".")

        # --- CONFIGURAÇÃO DO ÍCONE DA JANELA ---
        caminho_ico = os.path.join(caminho_base, "BioScanIco.ico")
        try:
            self.iconbitmap(caminho_ico)
        except Exception:
            pass

        # --- TELA DE CARREGAMENTO (OVERLAY) ---
        self.overlay = ctk.CTkFrame(self, fg_color="white")
        self.overlay.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.overlay_content = ctk.CTkFrame(self.overlay, fg_color="transparent")
        self.overlay_content.place(relx=0.5, rely=0.5, anchor="center")

        # --- ADIÇÃO DA LOGO ESTÁTICA ---
        try:
            from PIL import Image
            # Tenta carregar a logo (ajuste a extensão se for .jpg)
            caminho_logo = os.path.join(caminho_base, "BioScan Logo.png")
            if not os.path.exists(caminho_logo):
                caminho_logo = os.path.join(caminho_base, "BioScan Logo.jpg")
            
            img_logo = Image.open(caminho_logo)
            foto_logo = ctk.CTkImage(light_image=img_logo, dark_image=img_logo, size=(566, 337))
            
            self.label_logo = ctk.CTkLabel(self.overlay_content, image=foto_logo, text="")
            self.label_logo.pack(pady=10)
        except Exception:
            ctk.CTkLabel(self.overlay_content, text="🧬", font=("Arial", 60)).pack(pady=10)

        ctk.CTkLabel
        self.label_status = ctk.CTkLabel(self.overlay_content, text="Inicializando Motores de Inteligência Artificial...", font=("Arial", 14))
        self.label_status.pack(pady=5)
        
        # --- BARRA DE PROGRESSO ANIMADA ---
        self.progress = ctk.CTkProgressBar(self.overlay_content, width=400, mode="indeterminate", progress_color="#2980b9")
        self.progress.pack(pady=20)
        self.progress.start()

        threading.Thread(target=self.carregar_motores_ai, daemon=True).start()

    def carregar_motores_ai(self):
        global torch, AutoTokenizer, EsmModel, cosine_similarity, FPDF, datetime
        try:
            import torch
            from sklearn.metrics.pairwise import cosine_similarity
            from fpdf import FPDF
            from datetime import datetime
            from transformers import AutoTokenizer, EsmModel
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)

            MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = EsmModel.from_pretrained(MODEL_NAME)
            
            self.after(0, self.finalizar_carregamento)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro Crítico", f"Falha ao carregar bibliotecas: {e}"))

    def finalizar_carregamento(self):
        self.overlay.destroy()
        self.setup_ui()

    def setup_ui(self):
        self.resultados_armazenados = []
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(pady=10, padx=10, fill="both", expand=True)
        self.tabview.add("Análise Viral")
        self.tabview.add("Gerenciar Banco")
        
        self.setup_aba_analise()
        self.setup_aba_banco()

    def gerar_assinatura(self, sequencia):
        inputs = self.tokenizer(sequencia, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def setup_aba_analise(self):
        parent = self.tabview.tab("Análise Viral")
        
        # --- NOVO MENU SUPERIOR (TOOLBAR) ---
        self.toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        self.toolbar.pack(pady=10, padx=20, fill="x")
        
        # Botões de Arquivo alinhados à esquerda
        ctk.CTkButton(self.toolbar, text="CARREGAR FASTA", command=lambda: self.abrir_arquivo(self.textbox), 
                      fg_color="#34495e", width=140, height=35).grid(row=0, column=0, padx=5)
        
        self.btn_pdf = ctk.CTkButton(self.toolbar, text="SALVAR PDF", command=self.exportar_pdf, 
                                     fg_color="#c0392b", state="disabled", width=120, height=35, font=("Arial", 12, "bold"))
        self.btn_pdf.grid(row=0, column=1, padx=5)

        # Menu de Seleção de Receptor no Centro/Direita
        self.menu_receptores = ctk.CTkOptionMenu(self.toolbar, values=["RECEPTORES"] + list(banco_receptores.keys()), 
                                                 width=250, height=35, fg_color="#2980b9")
        self.menu_receptores.grid(row=0, column=2, padx=20)
        
        # Botão Iniciar em destaque no Menu Superior
        ctk.CTkButton(self.toolbar, text="ANALISAR", command=self.processar, 
                      fg_color="#27ae60", width=120, height=35, font=("Arial", 13, "bold")).grid(row=0, column=3, padx=5)

        # Botão Limpar
        ctk.CTkButton(self.toolbar, text="NOVA ANALISE", command=self.resetar, 
                      fg_color="#7f8c8d", width=40, height=35).grid(row=0, column=4, padx=5)

        # Expande o espaço entre os botões de arquivo e o menu de seleção
        self.toolbar.grid_columnconfigure(2, weight=1)

        # --- ÁREA CENTRAL ---
        ctk.CTkLabel(parent, text="BioScan AI: Predição Viral", font=("Arial", 20, "bold")).pack(pady=(5, 5))

        # Entrada de Texto agora ocupa mais espaço
        self.textbox = ctk.CTkTextbox(parent, height=150, border_width=2)
        self.textbox.insert("1.0", "INSIRA A SEQUÊNCIA DE AMINOÁCIDOS DA PROTEINA VIRAL AQUI...")
        self.textbox.pack(pady=5, padx=30, fill="x")

        self.txt_status_final = ctk.CTkLabel(parent, text="", font=("Arial", 12, "italic"))
        self.txt_status_final.pack()

        # ÁREA DE RESULTADOS
        self.scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="white", border_width=1)
        self.scroll_frame.pack(pady=10, padx=30, fill="both", expand=True)
        
        self.header_table = ctk.CTkFrame(self.scroll_frame, fg_color="#f2f4f7", corner_radius=0)
        self.header_table.pack(fill="x", pady=(0, 5))
        
        ctk.CTkLabel(self.header_table, text="RECEPTOR ALVO", font=("Arial", 13, "bold"), text_color="#34495e").grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        ctk.CTkLabel(self.header_table, text="|", font=("Arial", 13), text_color="#bdc3c7").grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(self.header_table, text="AFINIDADE (%)", font=("Arial", 13, "bold"), text_color="#34495e").grid(row=0, column=2, sticky="nsew", padx=10, pady=8)
        self.header_table.grid_columnconfigure((0, 2), weight=1)

    def mostrar_resultados(self):
        for w in self.scroll_frame.winfo_children():
            if w != self.header_table: w.destroy()

        for r in self.resultados_armazenados:
            row_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(row_frame, text=r[0], font=("Courier New", 13, "bold"), anchor="e").grid(row=0, column=0, sticky="nsew", padx=(0, 15))
            ctk.CTkLabel(row_frame, text="|", font=("Arial", 14), text_color="#bdc3c7").grid(row=0, column=1, sticky="nsew")
            ctk.CTkLabel(row_frame, text=f"{r[1]:>8.2f}%", font=("Courier New", 13, "bold"), text_color="#2ecc71", anchor="w").grid(row=0, column=2, sticky="nsew", padx=(15, 0))
            row_frame.grid_columnconfigure((0, 2), weight=1)
            ctk.CTkFrame(self.scroll_frame, height=1, fg_color="#eeeeee").pack(fill="x", padx=40)
        
        self.txt_status_final.configure(text="Análise Concluída!", text_color="#27ae60")
        self.btn_pdf.configure(state="normal")

    def setup_aba_banco(self):
        parent = self.tabview.tab("Gerenciar Banco")
        split = ctk.CTkFrame(parent, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=10, pady=10)

        left = ctk.CTkFrame(split, width=300, border_width=1); left.pack(side="left", fill="both", expand=False, padx=5); left.pack_propagate(False)
        ctk.CTkLabel(left, text="CADASTRAR RECEPTOR", font=("Arial", 16, "bold")).pack(pady=15)
        self.entry_nome = ctk.CTkEntry(left, placeholder_text="Ex: Receptor SARS-CoV-2", width=250); self.entry_nome.pack(pady=10)
        self.entry_seq = ctk.CTkTextbox(left, width=250, height=250); self.entry_seq.pack(pady=5)
        ctk.CTkButton(left, text="SALVAR NO SISTEMA", command=self.adicionar_receptor, fg_color="#2980b9", width=220, height=40).pack(pady=20)

        right = ctk.CTkFrame(split, border_width=1); right.pack(side="right", fill="both", expand=True, padx=5)
        ctk.CTkLabel(right, text="BIBLIOTECA ATIVA", font=("Arial", 16, "bold")).pack(pady=15)
        self.scroll_banco = ctk.CTkScrollableFrame(right, fg_color="white"); self.scroll_banco.pack(pady=10, padx=10, fill="both", expand=True)
        self.atualizar_listas()

    def atualizar_listas(self):
        for w in self.scroll_banco.winfo_children(): w.destroy()
        for nome in sorted(banco_receptores.keys()):
            f = ctk.CTkFrame(self.scroll_banco, fg_color="#f8f9fa", corner_radius=8); f.pack(fill="x", pady=3, padx=5)
            ctk.CTkLabel(f, text=f"• {nome}", font=("Arial", 12)).pack(side="left", padx=15, pady=8)
        self.menu_receptores.configure(values=["TODOS O BANCO"] + sorted(list(banco_receptores.keys())))

    def adicionar_receptor(self):
        n, s = self.entry_nome.get().strip(), limpar_sequencia(self.entry_seq.get("1.0", "end-1c"))
        if n and len(s) > 10:
            banco_receptores[n] = s; self.atualizar_listas(); messagebox.showinfo("Sucesso", f"Receptor {n} adicionado!");
            self.entry_nome.delete(0, 'end'); self.entry_seq.delete("1.0", "end")

    def processar(self):
        entrada = self.textbox.get("1.0", "end-1c")
        seq_estudo = limpar_sequencia(entrada)
        if len(seq_estudo) < 10: 
            messagebox.showwarning("Aviso", "Sequência inválida ou muito curta."); return
        
        escolha = self.menu_receptores.get()
        self.btn_pdf.configure(state="disabled")
        self.txt_status_final.configure(text="Executando predição por redes neurais...", text_color="#2980b9")
        threading.Thread(target=self.rodar_analise_ia, args=(seq_estudo, escolha), daemon=True).start()

    def rodar_analise_ia(self, seq_estudo, escolha):
        v_estudo = self.gerar_assinatura(seq_estudo)
        resultados = []
        alvos = banco_receptores.items() if escolha == "TODOS O BANCO" else [(escolha, banco_receptores[escolha])]
        for nome, s_rec in alvos:
            v_rec = self.gerar_assinatura(s_rec)
            score = float(cosine_similarity(v_estudo, v_rec)[0][0] * 100)
            resultados.append((nome, score))
        resultados.sort(key=lambda x: x[1], reverse=True)
        self.resultados_armazenados = resultados
        self.after(0, self.mostrar_resultados)

    def abrir_arquivo(self, target):
        caminho = filedialog.askopenfilename(filetypes=[("Arquivos FASTA", "*.fasta *.txt")])
        if caminho:
            with open(caminho, 'r') as f:
                target.delete("1.0", "end"); target.insert("1.0", f.read())

    def resetar(self):
        self.textbox.delete("1.0", "end")
        for w in self.scroll_frame.winfo_children():
            if w != self.header_table: w.destroy()
        self.btn_pdf.configure(state="disabled"); self.txt_status_final.configure(text="")

    def exportar_pdf(self):
        local = filedialog.asksaveasfilename(defaultextension=".pdf")
        if local:
            try:
                pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 15, "BioScan AI - Relatório de Afinidade Molecular", ln=True, align='C')
                pdf.ln(10); pdf.set_font("Courier", "B", 12)
                pdf.cell(140, 10, "RECEPTOR", border=1); pdf.cell(40, 10, "AFINIDADE", border=1, ln=True)
                pdf.set_font("Courier", "", 12)
                for r in self.resultados_armazenados:
                    pdf.cell(140, 10, f" {r[0]}", border=1); pdf.cell(40, 10, f" {r[1]:.2f}%", border=1, ln=True)
                pdf.output(local); messagebox.showinfo("Sucesso", "Relatório exportado!")
            except Exception as e: messagebox.showerror("Erro", str(e))

if __name__ == "__main__":
    app = BioScanApp()
    app.mainloop()