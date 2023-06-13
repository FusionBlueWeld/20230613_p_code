import os
import glob
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from janome.tokenizer import Tokenizer
import pdfplumber
import re
from sklearn.metrics.pairwise import cosine_similarity

class PatentProcessor:
    def __init__(self):
        # このファイルのパスを取得
        self.this_file_path = os.path.abspath(__file__)
        
        # このファイルが格納されているsrcフォルダのパスを取得
        self.src_folder_path = os.path.dirname(self.this_file_path)
        
        # srcフォルダの上位階層のrootフォルダのパスを取得
        self.root_folder_path = os.path.dirname(self.src_folder_path)
        
        # rootフォルダ内の、input_dataフォルダと、output_dataフォルダのパスを取得
        self.input_folder_path = os.path.join(self.root_folder_path, 'input_data')
        self.output_folder_path = os.path.join(self.root_folder_path, 'output_data')
        
        # 形態素解析用のオブジェクト生成
        self.tokenizer = Tokenizer()

    def tokenize(self, text):
        return [token.surface for token in self.tokenizer.tokenize(text)]

    def split_sections(self, text):
        # 正規表現パターンの定義
        pattern = r'(【(?=.*?[^\d０-９】])[^】]*?】)'
        
        # 正規表現パターンに基づいてテキストを分割
        sections = re.split(pattern, text)
        sections = sections[1:]  # 分割パターンの前に空文字列がある場合があるので、それを削除
        
        # セクション名と内容を辞書形式に格納
        sections_dict = {}
        for i in range(0, len(sections)-1, 2):
            sections_dict[sections[i]] = sections[i+1]

        # セクション名と内容の辞書を返す
        return sections_dict

    def process_patents(self):
        # globを用いて、フォルダ内のすべてのPDFファイルを取得
        pdf_files = glob.glob(os.path.join(self.input_folder_path, '*.pdf'))

        for pdf_file in pdf_files:
            # PDFを、テキストデータとして読み込み
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([p.extract_text() for p in pdf.pages])
            
            # 提案してくれた、明細書の区切りプログラムを、クラス内の関数として定義
            sections_dict = self.split_sections(text)

            # gensimのTaggedDocumentオブジェクトを作成
            documents = [TaggedDocument(words=self.tokenize(doc), tags=[i]) for i, doc in enumerate(sections_dict.values())]

            model_path = os.path.join(self.output_folder_path, 'patent_processing.model')
            
            # モデルの存在確認と追加学習
            if os.path.exists(model_path):
                # 既存のモデルを読み込み
                model = Doc2Vec.load(model_path)
                
                # モデルの語彙を更新
                model.build_vocab(documents, update=True)
                print(f"ロード：{pdf_file}")
            else:
                # モデルを新規に作成
                model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
                print(f"新規作成：{pdf_file}")

            # モデルを学習
            model.train(documents, total_examples=model.corpus_count, epochs=10)

            # モデルを保存
            model.save(model_path)

class PatentEvaluator:
    def __init__(self):
        # このファイルのパスを取得
        self.this_file_path = os.path.abspath(__file__)

        # このファイルが格納されているsrcフォルダのパスを取得
        self.src_folder_path = os.path.dirname(self.this_file_path)

        # srcフォルダの上位階層のrootフォルダのパスを取得
        self.root_folder_path = os.path.dirname(self.src_folder_path)

        # 学習済みモデルのパス
        self.model_path = os.path.join(self.root_folder_path, 'output_data', 'patent_processing.model')

        # rootフォルダ内の、input_data/test_dataフォルダのパスを取得
        self.test_folder_path = os.path.join(self.root_folder_path, 'input_data', 'test_data')

        # 形態素解析用のオブジェクト生成
        self.tokenizer = Tokenizer()

        # 学習済みモデルのロード
        self.model = Doc2Vec.load(self.model_path)

        print(self.model_path)

    def tokenize(self, text):
        return [token.surface for token in self.tokenizer.tokenize(text)]

    def split_sections(self, text):
        # 正規表現パターンの定義
        pattern = r'(【(?=.*?[^\d０-９】])[^】]*?】)'

        # 正規表現パターンに基づいてテキストを分割
        sections = re.split(pattern, text)
        sections = sections[1:]  # 分割パターンの前に空文字列がある場合があるので、それを削除

        # セクション名と内容を辞書形式に格納
        sections_dict = {}
        for i in range(0, len(sections)-1, 2):
            sections_dict[sections[i]] = sections[i+1]

        # セクション名と内容の辞書を返す
        return sections_dict

    def evaluate_patents(self):
        # globを用いて、フォルダ内のすべてのPDFファイルを取得
        pdf_files = glob.glob(os.path.join(self.test_folder_path, '*.pdf'))

        for pdf_file in pdf_files:
            # PDFを、テキストデータとして読み込み
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([p.extract_text() for p in pdf.pages])

            # 明細書の区切りプログラムを用いてテキストを分割
            sections_dict = self.split_sections(text)

            # 各セクションについて最大類似度の辞書を作成
            max_similarity_dict = {}

            # gensimのTaggedDocumentオブジェクトを作成
            for i, doc in enumerate(sections_dict.values()):
                tokens = self.tokenize(doc)
                vector = self.model.infer_vector(tokens)

                # すべてのベクトルとのコサイン類似度を計算
                similarity = cosine_similarity([vector], self.model.docvecs.vectors)

                # 最高値を保存
                max_similarity_dict[i] = max(similarity[0])

            print(f"PDF: {pdf_file}")
            for k, v in max_similarity_dict.items():
                print(f"Section {k}: Max similarity {v}")

def main():
    # processor = PatentProcessor()
    # processor.process_patents()

    evaluator = PatentEvaluator()
    evaluator.evaluate_patents()

if __name__ == '__main__':
    main()