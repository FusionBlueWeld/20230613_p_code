import os
import glob
import re
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sudachipy import Dictionary
import pdfplumber


class BasePatent:
    def __init__(self, model_path=None):
        # このファイルの絶対パスを取得
        self.this_file_path = os.path.abspath(__file__)
        # このファイルが存在するディレクトリ（srcフォルダ）のパスを取得
        self.src_folder_path = os.path.dirname(self.this_file_path)
        # srcフォルダの親ディレクトリ（rootフォルダ）のパスを取得
        self.root_folder_path = os.path.dirname(self.src_folder_path)
        # input_dataフォルダとoutput_dataフォルダのパスを取得
        self.input_folder_path = os.path.join(self.root_folder_path, 'input_data')
        self.output_folder_path = os.path.join(self.root_folder_path, 'output_data')

        # 学習済みのDoc2Vecモデルを読み込む（存在する場合）
        if model_path:
            self.model = Doc2Vec.load(model_path)

        # 形態素解析器のインスタンスを生成
        self.tokenizer = Dictionary().create()

    # 形態素解析を行うメソッド
    def tokenize(self, text):
        return [m.surface() for m in self.tokenizer.tokenize(text)]

    # テキストをセクションごとに分割するメソッド
    def split_sections(self, text):
        # 正規表現パターンを定義（【】で囲まれた部分をセクションとして分割）
        pattern = r'(【(?=.*?[^\d０-９】])[^】]*?】)'
        # パターンに基づいてテキストを分割
        sections = re.split(pattern, text)
        # 先頭の空文字列を削除（存在する場合）
        sections = sections[1:]
        # セクション名と内容を辞書形式に格納
        sections_dict = {}
        for i in range(0, len(sections)-1, 2):
            sections_dict[sections[i]] = sections[i+1]
        # セクション名と内容の辞書を返す
        return sections_dict


class PatentProcessor(BasePatent):
'''
このクラスは特許文書を前処理し、特許文書から学習用のデータを生成します。
具体的には、特許文書をセクションに分割し、各セクションを形態素解析して単語に分け、
その単語群をDoc2Vecモデルに入力して学習します。学習済みのモデルは指定のパスに保存されます。
'''
    def __init__(self):
        super().__init__()

    # 特許のプリプロセスを行うメソッド
    def preprocess_patents(self):
        # inputフォルダ内のすべてのPDFファイルを取得
        pdf_files = glob.glob(os.path.join(self.input_folder_path, '*.pdf'))

        for pdf_file in pdf_files:
            # PDFをテキストデータとして読み込み
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([p.extract_text() for p in pdf.pages])

            # テキストをセクションごとに分割
            sections_dict = self.split_sections(text)

            # すべてのセクションを前処理し、TokenizedDocumentのリストを生成
            documents = [TaggedDocument(words=self.tokenize(doc), tags=[i]) for i, doc in enumerate(sections_dict.values())]

            # gensimのDoc2Vecモデルを学習
            model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, epochs=40)

            # モデルを保存するパスを生成
            model_file = os.path.join(self.output_folder_path, 'patent_processing.model')

            # モデルを保存
            model.save(model_file)


class PatentEvaluator(BasePatent):
'''
このクラスは特許文書の評価を行います。
具体的には、新たに与えられた特許文書をセクションに分割し、各セクションを形態素解析して単語に分け、
その単語群を既存のDoc2Vecモデルに入力してベクトルを生成します。
それから、生成したベクトルと学習済みモデル内の全ベクトルとの間のコサイン類似度を計算し、最大の類似度を出力します。
'''
    def __init__(self, model_path):
        super().__init__()

        # 学習済みのDoc2Vecモデルを読み込む
        self.model = Doc2Vec.load(model_path)

    # 特許の評価を行うメソッド
    def evaluate_patents(self):
        # テストデータのフォルダ内のすべてのPDFファイルを取得
        pdf_files = glob.glob(os.path.join(self.test_folder_path, '*.pdf'))

        for pdf_file in pdf_files:
            # PDFをテキストデータとして読み込み
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([p.extract_text() for p in pdf.pages])

            # テキストをセクションごとに分割
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



class PatentVectorizer(BasePatent):
'''
このクラスは特許文書をベクトル化し、それをファイルとして出力します。
具体的には、特許文書をセクションに分割し、各セクションを形態素解析して単語に分け、
その単語群を既存のDoc2Vecモデルに入力してベクトルを生成します。
各セクションのベクトルを結合して2次元ベクトルを生成し、それをCSVファイルとして保存します。
'''
    def __init__(self, model_path):
        super().__init__()

        # 学習済みのDoc2Vecモデルを読み込む
        self.model = Doc2Vec.load(model_path)

    # 特許をベクトル化するメソッド
    def vectorize_patents(self):
        # 入力データのフォルダ内のすべてのPDFファイルを取得
        pdf_files = glob.glob(os.path.join(self.input_folder_path, '*.pdf'))

        for pdf_file in pdf_files:
            # PDFをテキストデータとして読み込み
            with pdfplumber.open(pdf_file) as pdf:
                text = '\n'.join([p.extract_text() for p in pdf.pages])

            # テキストをセクションごとに分割
            sections_dict = self.split_sections(text)

            # すべてのセクションのベクトルを保存するためのリスト
            vectors = []

            # セクションごとにベクトル化
            for doc in sections_dict.values():
                tokens = self.tokenize(doc)
                vector = self.model.infer_vector(tokens)
                vectors.append(vector)
            
            # すべてのセクションのベクトルを結合して2次元ベクトルを生成
            document_vector = np.vstack(vectors)

            # PDFファイル名から拡張子を削除し、それをフォルダ名として使用
            folder_name = os.path.splitext(os.path.basename(pdf_file))[0]
            output_folder = os.path.join(self.output_folder_path, folder_name)

            # フォルダが存在しなければ作成
            os.makedirs(output_folder, exist_ok=True)

            # CSVファイルのパスを生成
            csv_file = os.path.join(output_folder, 'vector.csv')

            # 2次元ベクトルをCSVファイルに出力
            np.savetxt(csv_file, document_vector, delimiter=',')



def main():
    model_path = "path_to_model"
    
    # Process patents
    processor = PatentProcessor(model_path)
    processor.process_patents()
    
    # Evaluate patents
    evaluator = PatentEvaluator(model_path)
    evaluator.evaluate_patents()
    
    # Vectorize patents
    vectorizer = PatentVectorizer(model_path)
    vectorizer.vectorize_patents()


if __name__ == "__main__":
    main()
