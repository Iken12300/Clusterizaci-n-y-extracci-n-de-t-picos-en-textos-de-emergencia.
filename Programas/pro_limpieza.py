"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Trabajo de Luis Guerrero y Fabian Muñoz
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Librerias
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
import pandas as pd
import re

"""nltk.download('stopwords')
nltk.download('punkt')"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Procesos de limpieza y preparacion (p1)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
def p1_limpieza_regex(conversacion):
    pr1 = re.sub(r'[#\-}¥ø]', '', conversacion)
    pr2 = re.sub(r'(\b\w+\b)(\W+\1)+', r'\1', pr1)
    pr3 = re.sub(r'\b(\w+)( \1\b)+', r'\1', pr2)
    pr4 = re.sub(r'\w*\d\w*', '', pr3)
    return pr4

def p1_limpieza_stopwords(dataset):
    stop_words = set(stopwords.words('spanish'))
    dataset['ATA_TEXTO'] = dataset['ATA_TEXTO'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    return dataset

def p1_limpieza_palabras(conversacion):
    palabras_adicionales = ['u', 'i', 'h', 'j', 'l', 'v', 
                            'b', 's', 'p', 'J', 'n', 'ñ', 
                            'c', 't', 'f', 'r', 'd', 
                            'm', 'z', 'si', 'eh', 'va', 
                            'no', 'es', 'su', 'un', 'uh', 
                            'um', 'eh', 'ah', 'ok', 'sí', 'no', 
                            'em', 'ya', 'hi', 'di', 'ne', 'ex', 
                            'da', 'oh', 'uy', 'as', 'ja', 'jr', 
                            'ar', 'yy', 'ee', 'ca', 'sq', 
                            've', 'qe', 'ta', 'gh', 'in', 'bi', 
                            'ir', 'cz', 'fq', 'ye', 'jo', 'to',
                            'sp', 'ps', 'up', 'ab', 'hm', 'ea', 
                            'is', 'sz', 'dj', 'pj', 'so', 'sa', 
                            'ma', 'pa', 'sh', 'ur', 'lb', 'an', 
                            'Es', 'bt', 'ay', 'dé', 'aa', 'av', 
                            'td', 'mm', 'am', 'co', 'jb', 'ud', 
                            'po', 'ls', 'cu', 'fa', 'ahi', 'mas', 
                            'alo', 'ecu', 'aca', 'ahí', 'san', 
                            'dos', 'acá', 'asi', '911', 'muy', 'mmm', 
                            'shh', 'oye', 'hmm', 'ehm', 'ajá', 'ahá', 
                            'mmm', 'ahí', 'así', 'car', 'aaa', 'sur', 
                            'eme', 'sub', 'feo', 'nnn', 'per', 'haz', 
                            'ham', 'ara', 'wal', 'abf', 'tlo', 'dan', 
                            'pai', 'uso', 'dle', 'tbi', 'ska', 'pur', 
                            'pro', 'men', 'tez', 'yde', 'sht', 'ape', 
                            'abo', 'cua', 'ach', 'cas', 'bxw', 'laa', 
                            'vis', 'jue', 'ule', 'lis', 'aha', 'lbp', 
                            'ami', 'aba', 'pan', 'dug', 'ola', 'Alo', 
                            'yap', 'cab', 'pol', 'dis', 'lay', 'fea', 
                            'lei', 'ñor', 'pxa', 'ben', 'cre', 'acg', 
                            'adh', 'yaa', 'nin', 'icn', 'mju', 'lie', 
                            'fum', 'hac', 'yyy', 'tbb', 'crs', 'tlu', 
                            'aau', 'lla', 'eee', 'fuu', 'des', 'poe', 
                            'djo', 'enn', 'lut', 'cue', 'abc', 'ehh', 
                            'hut', 'fre', 'kia', 'spa', 'gas', 'tik', 
                            'see', 'val', 'min', 'cha', 'ade', 'hom', 
                            'ups', 'une', 'plo', 'dal', 'qui', 'dee', 
                            'etc', 'ahh', 'dar', 'tcz', 'esl', 'rcp', 
                            'eno', 'lui', 'uah', 'abk', 'red', 'ehe', 
                            'afk', 'rav', 'aui', 'tea', 'nse', 'llo', 
                            'roo', 'ant', 'tom', 'mhm', 'upc', 'cme', 
                            'tre', 'ele', 'cte', 'duy', 'hur', 'acd', 
                            'qeu', 'hee', 'eje', 'een', 'wel', 'die', 
                            'pbf', 'tob', 'lbt', 'afp', 'pob', 'toy', 
                            'abb', 'toa', 'tbt', 'sii', 'cho', 'luv', 
                            'vaa', 'med', 'gua', 'aqui', 'aquí', 'sabe', 
                            'bien', 'años', 'deme', 'dice', 'oiga', 'lado', 
                            'pasa', 'dias', 'días', 'vera', 'creo', 'hace', 
                            'unas', 'tres', 'toda', 'paso', 'como', 'cómo', 
                            'este', '¡Ay!', '¡Oh!', 'bang', 'pues', 'vale', 
                            'okey', 'vaya', 'aquí', 'allí', 'señor', 'favor', 
                            'estan', 'listo', 'usted', 'luego', 'ayuda', 'justo', 
                            'menos', 'mande', 'puede', 'hacer', 'puedo', 'abajo', 
                            'bueno', 'medio', 'gente', 'mismo', 'quera', 'bueno', 
                            'crack', 'claro', 'sí sí', 'a ver', 'ya ya', 'luego', 
                            'quizá', 'nombre', 'muchas', 'señora', 'perdon', 'buenas', 
                            'buenos', 'hagame', 'tardes', 'sector', 'ayudar', 'hágame', 
                            'arriba', 'perdón', 'amable', 'malito', 'mandar', 'quiere', 
                            'pueden', 'noches', 'queria', 'quería', 'cierto', 'ya veo', 
                            'exacto', 'además', 'quizás', 'gracias', 'ayudeme', 'ayúdeme', 
                            'cuantos', 'ahorita', 'entrada', 'señores', 'senñora', 'pasaria', 
                            'pasaría', 'tal vez', 'gracias', 'así que', 'oh vaya', '¿sabes?', 
                            'también', 'por eso', 'después', 'tal vez', 'coordino', 'haciendo', 
                            'entonces', 'llamando', 'porfavor', 'disculpe', 'necesito', 'personas', 
                            'senñores', 'acabamos', 'disculpe', 'entonces', 'sí claro', 'perfecto', 
                            'de hecho', 'entiendo', 'en serio', 'entonces', 'ahí está', 'es decir', 
                            'entonces', 'entonces', 'entonces', 'da igual', 'saludamos', 'ingresado', 
                            'por favor', 'entendido', 'entendido', 'lo siento', 'pues pues', 'puede ser', 
                            'de pronto', 'emergencia', '¡Dios mío!', 'de acuerdo', 'ahora bien', 'déjame ver', 
                            'obviamente', 'de repente', 'a lo mejor', 'lo que sea', 'coordinamos', 'lo que pasa', 
                            'bueno bueno', 'me refiero a', 'quiero decir', 'por supuesto', 'naturalmente', 'claro que sí', 
                            'por lo tanto', 'posiblemente', 'como quieras', 'lo que sucede', 'déjame pensar', 'evidentemente', 
                            'probablemente', 'lo que tú digas', 'por consiguiente', 'lo que quiero decir es']
    palabras = conversacion.split()
    palabras = [palabra for palabra in palabras if palabra not in palabras_adicionales]
    conversacion_corregida = ' '.join(palabras)
    return conversacion_corregida

def p1_preparacion(dataset):
    # Eliminar las columnas especificadas
    dataset.drop(columns=["INCIDENTGRADENAME", "ATA_ACTOR", "ATA_SECUENCIA"], inplace=True)
    
    # Unir todas las filas que posean el mismo id en la columna 'TRA_ID'
    # Aqui se pierde la columna 'TRA_ID'
    dataset = dataset.astype(str).groupby('TRA_ID').agg(lambda x: ' '.join(x))
    
    # Rescatar todos los caracteres del dataset
    #caracteres = "".join(sorted(set(" ".join(dataset.astype(str).values.flatten()))))
    #print(caracteres)

    # Limpiar los caracteres erroneos
    dataset['ATA_TEXTO'] = dataset['ATA_TEXTO'].apply(p1_limpieza_regex)

    # Limpiar las stopwords
    dataset = p1_limpieza_stopwords(dataset)

    # Eliminar palabras irrelevantes
    dataset['ATA_TEXTO'] = dataset['ATA_TEXTO'].apply(p1_limpieza_palabras)

    # Eliminar los campos vacios
    dataset['ATA_TEXTO'] = dataset['ATA_TEXTO'].replace('', pd.NA)
    dataset.dropna(subset=['ATA_TEXTO'], inplace=True)
    return dataset
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Proceso para rescatar todas las palabras de la base de datos (p2)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
def p2_total_palabras(dataset):
    dataset['palabras'] = dataset['ATA_TEXTO'].str.split()
    palabs_base = [palabra for lista in dataset['palabras'] for palabra in lista]
    palabs_base = list(set(palabs_base))
    return palabs_base
"""
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Main
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
if __name__ == "__main__":
    # Lectura de los datos
    dataset = pd.read_csv(r"C:/Users/Luis/Desktop/gestco_examen/3_programa/dataset.csv", sep=";", on_bad_lines="warn")

    dataset = p1_preparacion(dataset)
    print(dataset.shape)
    print(dataset.head())

    """lista = p2_total_palabras(dataset)
    lista = sorted(lista, key=len)
    print(lista)"""

    dataset.to_csv("C:/Users/Luis/Desktop/gestco_examen/3_programa/conversaciones.csv", index=False)
