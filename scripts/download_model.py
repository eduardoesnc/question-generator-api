"""
Script para baixar o modelo spaCy durante o deploy
"""
import subprocess
import sys

def download_spacy_model():
    """Baixa o modelo pt_core_news_sm do spaCy"""
    try:
        print("Baixando modelo spaCy pt_core_news_sm...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "spacy", 
            "download", 
            "pt_core_news_sm"
        ])
        print("✅ Modelo baixado com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao baixar modelo: {e}")
        return False

if __name__ == "__main__":
    success = download_spacy_model()
    sys.exit(0 if success else 1)
