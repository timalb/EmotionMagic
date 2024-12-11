import os
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from gigachat import GigaChat
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "harry-potter-magic-key"

# Initialize GigaChat client with SSL verification disabled for development
gigachat_client = GigaChat(
    credentials=os.environ.get('GIGACHAT_API_KEY'),
    verify_ssl_certs=False  # Only for development
)

def analyze_text_with_gigachat(text):
    """Analyze text using GigaChat to extract personality traits and key phrases."""
    try:
        prompt = f"""Проанализируй следующий текст с точки зрения распределяющей шляпы из Гарри Поттера.
        Определи черты характера и ключевые фразы, которые указывают на принадлежность к факультетам.
        Текст: "{text}"
        
        Дай ответ в формате JSON:
        {{
            "key_phrases": [список важных цитат из текста],
            "traits": [обнаруженные черты характера],
            "analysis": [краткие выводы по тексту]
        }}"""
        
        logging.info(f"Sending request to GigaChat API with text: {text[:100]}...")
        
        try:
            response = gigachat_client.chat(prompt)
            logging.info(f"Raw API response: {response}")
            
            if not hasattr(response, 'choices') or not response.choices:
                logging.error("Invalid response format: no choices found")
                raise ValueError("Invalid API response format")
                
            content = response.choices[0].message.content
            logging.info(f"Response content: {content[:500]}")
            
            try:
                parsed_response = json.loads(content)
                logging.info(f"Successfully parsed JSON response: {parsed_response}")
                
                # Validate response structure
                if not all(key in parsed_response for key in ['key_phrases', 'traits', 'analysis']):
                    logging.error(f"Missing required keys in response: {parsed_response}")
                    raise ValueError("Invalid response structure")
                
                return parsed_response
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {str(e)}")
                # If JSON parsing fails, try to extract meaningful content
                return {
                    "key_phrases": [],
                    "traits": [],
                    "analysis": [content] if content else ["Ответ получен, но его формат не соответствует ожидаемому"]
                }
                
        except Exception as api_error:
            logging.error(f"API call error: {str(api_error)}")
            raise
            
    except Exception as e:
        logging.error(f"Critical error in GigaChat analysis: {str(e)}")
        logging.exception("Full traceback:")
        return {
            "key_phrases": [],
            "traits": [],
            "analysis": ["Не удалось провести глубокий анализ текста. Пожалуйста, попробуйте позже."]
        }

def generate_sorting_hat_monologue(scores, text, gigachat_analysis):
    """Generate a rich, personalized Harry Potter style monologue with varied magical stories."""
    from random import choice, random, sample
    
    house_translations = {
        'Gryffindor': 'Гриффиндор',
        'Hufflepuff': 'Пуффендуй',
        'Ravenclaw': 'Когтевран',
        'Slytherin': 'Слизерин'
    }

    # Магические артефакты для разнообразия описаний
    magical_artifacts = [
        "древний фолиант", "магический кристалл", "зачарованное зеркало",
        "философский камень", "песочные часы судьбы", "карта звёздного неба",
        "волшебный глобус", "магическая призма", "чаша провидения",
        "астрологический компас", "руны древних магов", "сфера пророчеств"
    ]

    # Магические существа для метафор
    magical_creatures = [
        "феникс", "единорог", "гиппогриф", "дракон",
        "василиск", "кентавр", "русалка", "мантикора",
        "нюхлер", "вейла", "гном", "домовой эльф"
    ]

    # Магические места
    magical_places = [
        "Запретный лес", "Выручай-комната", "Астрономическая башня",
        "подземелья Хогвартса", "Большой зал", "берег Чёрного озера",
        "совятня", "тайная комната", "библиотека Хогвартса"
    ]
    
    sorted_houses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_house = sorted_houses[0][0]
    top_house_name = house_translations[top_house]
    second_house_name = house_translations[sorted_houses[1][0]]
    
    # Разнообразные варианты начальных фраз
    openings = [
        f"*{choice(magical_artifacts)} начинает светиться, когда я касаюсь твоего разума*\nОх! Какие необычные мысли...",
        f"*древние руны на моих полях вспыхивают, словно {choice(magical_creatures)} пробудился ото сна*\nКак интересно...",
        f"*магия {choice(magical_places)} отзывается на твои мысли*\nДавно не встречала такого своеобразного сочетания качеств...",
        "*тысячелетняя пыль магии кружится вокруг*\nТвой разум... он напоминает мне древний лабиринт...",
        f"*{choice(magical_artifacts)} в кабинете директора реагирует на твоё присутствие*\nКакой необычный узор мыслей...",
        "*старинные портреты директоров затихают, прислушиваясь*\nВ твоём сознании я вижу нечто особенное...",
        f"*магические свитки в {choice(magical_places)} трепещут от предвкушения*\nТвой разум похож на редкий магический кристалл..."
    ]

    # Формируем уникальное начало монолога
    chosen_artifact = choice(magical_artifacts)
    chosen_creature = choice(magical_creatures)
    chosen_place = choice(magical_places)
    
    custom_opening = (
        f"*{chosen_artifact} мерцает загадочным светом, когда я погружаюсь в твои мысли*\n"
        f"Подобно {chosen_creature} в {chosen_place}, твой разум полон неожиданных поворотов..."
    )
    
    # Случайный выбор между стандартным и уникальным началом
    monologue = [choice([choice(openings), custom_opening])]

    # Добавляем случайную деталь о магической атмосфере
    atmospheric_details = [
        f"*{choice(magical_artifacts)} вибрирует от магической энергии*",
        f"*в воздухе появляются искры, напоминающие созвездие {choice(['Дракона', 'Феникса', 'Единорога', 'Гидры'])}*",
        f"*эхо древних заклинаний разносится по {choice(magical_places)}*",
        f"*{choice(magical_creatures)} где-то вдалеке отзывается на магию момента*"
    ]
    monologue.append(choice(atmospheric_details))
    
    # Глубокий анализ ключевых фраз с магическими историями
    if gigachat_analysis.get('key_phrases'):
        phrases = gigachat_analysis['key_phrases']
        if phrases:
            phrase = choice(phrases)
            # Создаём уникальную историю для каждой фразы
            magical_events = [
                "Битва Четырёх Стихий", "Турнир Древних Заклинаний",
                "Собрание Великого Совета", "Ритуал Пробуждения Магии",
                "Церемония Звёздного Света", "Испытание Элементалей"
            ]
            famous_spells = [
                "Люмос Максима", "Протего Тоталум", "Экспекто Патронум",
                "Приори Инкантатем", "Фиделиус", "Пиротехника Уизли"
            ]
            magical_books = [
                "Хроники Забытой Магии", "Тайны Древних Волшебников",
                "Пророчества Тысячелетий", "Легенды Хогвартса",
                "Секреты Мерлина", "Записки Основателей"
            ]

            # Генерируем уникальную историю
            magical_interpretations = [
                f"*{choice(magical_artifacts)} оживает* Твои слова '{phrase}' – они как эхо {choice(magical_events)}! В тот день {choice(magical_creatures)} был свидетелем великой магии...",
                
                f"*древние руны складываются в узоры* В книге '{choice(magical_books)}' я читала о волшебнике, который произнёс '{phrase}' перед тем, как создать заклинание '{choice(famous_spells)}'...",
                
                f"*магический туман формирует видения* Знаешь, в {choice(magical_places)} до сих пор хранится память о словах '{phrase}'. Их произнёс {choice(['юный ученик', 'древний маг', 'загадочный волшебник'])}, когда впервые открыл тайну {choice(magical_artifacts)}...",
                
                f"*{choice(magical_creatures)} появляется в магическом мареве* Твоя фраза '{phrase}' пробуждает древнее пророчество! Оно гласит, что когда эти слова будут произнесены в {choice(magical_places)}, начнётся новая эпоха магии...",
                
                f"*{choice(magical_artifacts)} испускает загадочное сияние* В твоих словах '{phrase}' я слышу отголоски {choice(magical_events)}. Тогда {choice(magical_creatures)} указал путь к {choice(magical_places)}, где хранился древний секрет...",
                
                f"*старинные фолианты сами открываются* О! '{phrase}' – эти слова начертаны на страницах '{choice(magical_books)}' рядом с описанием {choice(famous_spells)}. Легенда гласит, что они способны {choice(['пробуждать древнюю магию', 'открывать тайные двери', 'призывать магических существ'])}..."
            ]
            
            # Добавляем несколько интерпретаций для создания более богатого повествования
            monologue.extend(sample(magical_interpretations, k=min(3, len(magical_interpretations))))
            
            # Добавляем дополнительную интерпретацию для другой фразы, если она есть
            if len(phrases) > 1:
                other_phrase = choice([p for p in phrases if p != phrase])
                additional_interpretations = [
                    f"*искры магии вспыхивают* А эти слова - '{other_phrase}' - несут в себе особую силу. Они напоминают мне заклинание, которое...",
                    f"*древние руны светятся* Интересно... '{other_phrase}' - эта фраза появляется в старинной книге предсказаний...",
                    f"*магический компас начинает вращаться* Твоё '{other_phrase}' указывает на необычный путь развития твоей магии..."
                ]
                monologue.append(choice(additional_interpretations))
    
    # Добавляем персонализированные истории на основе черт характера
    if gigachat_analysis.get('traits'):
        traits = gigachat_analysis['traits']
        if traits:
            trait = choice(traits).lower()
            stories = [
                f"*погружается в воспоминания* Знаешь, твоя {trait} напомнила мне об одном юном волшебнике...",
                f"*магическая дымка окутывает шляпу* Твоя {trait} похожа на редкое магическое свойство...",
                f"*искры магии вспыхивают* О! Эта {trait} - прямо как у легендарного выпускника..."
            ]
            monologue.append(choice(stories))
    
    # Добавляем анализ в форме магических видений
    if gigachat_analysis.get('analysis'):
        analysis = gigachat_analysis['analysis'][0]
        if not analysis.startswith('{') and '```' not in analysis:
            visions = [
                "*в древней памяти шляпы всплывает видение*",
                "*магический туман формирует образы*",
                "*старинные чары раскрывают картину*"
            ]
            monologue.append(f"{choice(visions)} {analysis}")
    
    # Драматическое сравнение факультетов с магическими метафорами
    house_metaphors = {
        'Гриффиндор': ['пламя феникса', 'рык льва', 'красно-золотые искры'],
        'Пуффендуй': ['тепло солнечного света', 'аромат волшебных трав', 'мерцание янтаря'],
        'Когтевран': ['полёт мудрой совы', 'сияние звёздного неба', 'шелест древних страниц'],
        'Слизерин': ['шепот змеи', 'блеск изумрудов', 'таинственные глубины озера']
    }
    
    monologue.append(
        f"*магический вихрь закручивается* В тебе я вижу {choice(house_metaphors[top_house_name])}... "
        f"Но что это? {choice(house_metaphors[second_house_name])} тоже проявляется..."
    )
    
    # Близкие результаты делают выбор более драматичным
    if abs(sorted_houses[0][1] - sorted_houses[1][1]) < 10:
        dramatic_moments = [
            f"*древняя магия колеблется между {top_house_name} и {second_house_name}*",
            "*шляпа погружается в глубокие раздумья*",
            "*магические энергии сплетаются в причудливый узор*"
        ]
        monologue.extend([choice(dramatic_moments)])
    
    # Финальное решение с расширенной магической историей
    key_traits = gigachat_analysis.get('traits', [])
    if key_traits:
        trait = choice(key_traits).lower()
        legendary_students = {
            'Гриффиндор': ['Годрик Гриффиндор', 'Альбус Дамблдор', 'Гарри Поттер'],
            'Пуффендуй': ['Хельга Пуффендуй', 'Ньют Саламандер', 'Седрик Диггори'],
            'Когтевран': ['Ровена Когтевран', 'Луна Лавгуд', 'Чжоу Чанг'],
            'Слизерин': ['Салазар Слизерин', 'Северус Снейп', 'Мерлин']
        }
        
        legendary_student = choice(legendary_students[top_house_name])
        endings = [
            f"*магический вихрь формирует образы* Твоя {trait} напоминает мне молодого {legendary_student}. В начале своего пути он тоже...",
            f"*древние страницы оживают* О! Эта {trait} - точь-в-точь как у {legendary_student} в его первый день в Хогвартсе...",
            f"*магическое зеркало показывает видение* Твой путь может быть похож на путь {legendary_student}. Та же {trait}, то же стремление...",
            f"*философский камень мерцает* В твоей {trait} я вижу отражение качеств {legendary_student}. Возможно, тебя ждёт похожая судьба...",
            f"*чернила на пергаменте складываются в узоры* Знаешь, {legendary_student} тоже отличался особой {trait}. И посмотри, каких высот он достиг!"
        ]
    else:
        endings = [
            "*магические символы кружатся в воздухе* В твоих словах я вижу отражение древней магии...",
            "*хрустальный шар наполняется туманом* Твой путь уникален, как редкое сочетание звёзд...",
            "*феникс пролетает над головой* Твоя судьба начинает проявляться в магическом пламени...",
            "*древние руны складываются в пророчество* Твоё будущее пишется прямо сейчас...",
            "*волшебные часы начинают идти вспять* В тебе есть что-то, что напоминает мне о великих временах..."
        ]
    
    monologue.extend([
        choice(endings),
        "О да... *магический вихрь успокаивается*",
        "Теперь я вижу это совершенно ясно...",
        "*финальная вспышка магии*",
        f"{top_house_name.upper()}!"
    ])
    
    return monologue

def adjust_text_sentiment(text, make_positive=True):
    """Adjust text sentiment by adding magical words."""
    positive_words = [
        "Люмос", "Экспекто Патронум", "Винградиум Левиоса",
        "Фелицис Фелицис", "Алохомора", "Репаро"
    ]
    negative_words = [
        "Нокс", "Морсмордре", "Круцио", 
        "Обливиэйт", "Империо", "Авада Кедавра"
    ]
    
    words = text.split()
    if make_positive:
        words.insert(0, positive_words[hash(text) % len(positive_words)])
    else:
        words.insert(0, negative_words[hash(text) % len(negative_words)])
    
    return " ".join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Get deep analysis from GigaChat
        gigachat_analysis = analyze_text_with_gigachat(text)
        
        # Traditional sentiment analysis
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Calculate house affinities based on both TextBlob and GigaChat analysis
        trait_scores = {
            'Gryffindor': 0,
            'Hufflepuff': 0,
            'Ravenclaw': 0,
            'Slytherin': 0
        }
        
        # Базовые очки для каждого факультета, чтобы избежать сильных перекосов
        base_points = 25
        trait_scores['Gryffindor'] = base_points + (max(sentiment_score, 0) * 0.5 + subjectivity * 0.5) * 25
        trait_scores['Hufflepuff'] = base_points + ((1 - abs(sentiment_score)) * 0.5 + subjectivity * 0.5) * 25
        trait_scores['Ravenclaw'] = base_points + ((1 - subjectivity) * 0.5 + (1 + sentiment_score) * 0.5) * 25
        trait_scores['Slytherin'] = base_points + ((1 + abs(sentiment_score)) * 0.5 + (1 - subjectivity) * 0.5) * 25
        
        # Add scores based on GigaChat analysis
        house_keywords = {
            'Gryffindor': ['храбрость', 'отвага', 'смелость', 'благородство', 'решительность'],
            'Hufflepuff': ['трудолюбие', 'верность', 'честность', 'доброта', 'справедливость'],
            'Ravenclaw': ['мудрость', 'ум', 'знания', 'творчество', 'логика'],
            'Slytherin': ['амбиции', 'хитрость', 'находчивость', 'цель', 'власть']
        }
        
        # Analyze traits from GigaChat
        for trait in gigachat_analysis['traits']:
            trait = trait.lower()
            for house, keywords in house_keywords.items():
                if any(keyword in trait for keyword in keywords):
                    trait_scores[house] += 25
        
        # Normalize scores to ensure they add up to 100%
        total_score = sum(trait_scores.values())
        if total_score > 0:
            houses = {
                house: round((score / total_score) * 100, 1)
                for house, score in trait_scores.items()
            }
        else:
            houses = {house: 25.0 for house in trait_scores.keys()}
        
        # Generate the sorting hat's monologue with GigaChat insights
        monologue = generate_sorting_hat_monologue(houses, text, gigachat_analysis)
        
        return jsonify({
            'houses': houses,
            'monologue': monologue,
            'final_house': max(houses.items(), key=lambda x: x[1])[0],
            'analysis': gigachat_analysis
        })
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return jsonify({'error': 'Магическая ошибка произошла!'}), 500

@app.route('/transform', methods=['POST'])
def transform_text():
    try:
        text = request.json.get('text', '')
        direction = request.json.get('direction', 'positive')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        transformed_text = adjust_text_sentiment(
            text, 
            make_positive=(direction == 'positive')
        )
        
        return jsonify({'text': transformed_text})
    except Exception as e:
        logging.error(f"Error in text transformation: {str(e)}")
        return jsonify({'error': 'Заклинание дало обратный эффект!'}), 500
