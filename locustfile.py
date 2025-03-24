import json
import random
import time

from locust import HttpUser, between, constant_pacing, task

french_paragraphs = [
    "L'objectif de Développement Durable numéro 1 est d'éliminer la pauvreté sous toutes ses formes et partout dans le monde. Cela comprend la réduction de la pauvreté extrême, la promotion de l'emploi décent et la mise en place de systèmes de protection sociale pour les plus vulnérables.",
    "Le deuxième objectif vise à éliminer la faim et assurer la sécurité alimentaire. Cela implique de promouvoir l'agriculture durable, de soutenir les petits agriculteurs et de garantir l'accès à une alimentation nutritive pour tous.",
    "L'objectif 3 est d'assurer une bonne santé et promouvoir le bien-être de tous à tous les âges. Cela signifie garantir l'accès à des services de santé de qualité, promouvoir la prévention des maladies et l'accès aux médicaments essentiels.",
    "Il est essentiel de fournir une éducation de qualité inclusive et équitable pour tous, l'objectif 4. Cela comprend la promotion de l'éducation primaire gratuite, l'accès à des formations professionnelles et l'élimination des disparités entre les genres dans l'éducation.",
    "L'objectif 5 cherche à parvenir à l'égalité des sexes et à autonomiser toutes les femmes et les filles. Cela nécessite la fin de la violence et de toutes les formes de discrimination à leur égard, ainsi que l'accès à des droits sexuels et reproductifs.",
    "Afin de garantir l'accès à l'eau potable et à l'assainissement pour tous, l'objectif 6 mise sur la gestion durable de cette ressource vitale. Il est nécessaire de réduire la pollution de l'eau, d'améliorer l'efficacité de son utilisation et de garantir l'accès à des installations sanitaires adéquates.",
    "L'objectif 7 consiste à garantir l'accès à une énergie propre, abordable et durable pour tous. Cela comprend le développement des énergies renouvelables, l'amélioration de l'efficacité énergétique et l'accès à des technologies énergétiques modernes.",
    "Le travail décent et la croissance économique durable sont l'objectif 8. Cela implique de promouvoir des emplois décents, de soutenir l'entrepreneuriat et de favoriser une croissance économique équitable et durable.",
    "L'objectif 9 vise à promouvoir l'industrialisation durable et l'innovation. Cela signifie la construction d'infrastructures résilientes, promouvoir l'innovation technologique et faciliter l'accès des pays en développement aux technologies de l'information et de la communication.",
    "L'objectif 10 cherche à réduire les inégalités au sein et entre les pays. Cela nécessite de promouvoir des politiques favorables à une croissance économique inclusive et de mettre en œuvre une régulation et une gestion financière soucieuses des besoins des pays en développement.",
    "Il est essentiel de rendre les villes et les établissements humains inclusifs, sûrs, résilients et durables. C'est l'objectif 11. Pour y parvenir, il est nécessaire d'investir dans des infrastructures durables, la planification urbaine et la gestion des déchets.",
    "L'objectif 12 consiste à garantir des modes de consommation et de production durables. Cela comprend la promotion de la consommation responsable, la réduction de la quantité de déchets produits et la gestion durable des ressources naturelles.",
    "La lutte contre le changement climatique et ses impacts est l'objectif 13. Cela nécessite des mesures pour réduire les émissions de gaz à effet de serre, promouvoir l'utilisation des énergies renouvelables et renforcer la résilience face aux changements climatiques.",
    "La préservation et la restauration des écosystèmes terrestres, l'objectif 15, sont essentiels pour la conservation de la biodiversité et la lutte contre l'érosion de la biodiversité. Cela implique la gestion durable des forêts, la lutte contre la déforestation et la protection des espèces menacées.",
    "L'objectif 16 vise à promouvoir des sociétés pacifiques, justes et inclusives. Cela nécessite de réduire la violence et les taux de criminalité, de promouvoir l'état de droit et l'accès à la justice pour tous.",
    "La mise en place de partenariats mondiaux efficaces pour le développement durable est l'objectif 17. Cela implique de renforcer la coopération internationale, la mobilisation des ressources financières et la technologie pour atteindre les Objectifs de Développement Durable.",
]

english_paragraphs = [
    "Sustainable Development Goal 1: No Poverty. Ending poverty in all its forms everywhere is the first goal of the UN's Sustainable Development Agenda. This includes ensuring that everyone has access to basic services, social protection, and economic opportunities.",
    "Sustainable Development Goal 2: Zero Hunger. Achieving food security, improving nutrition, and promoting sustainable agriculture are all key components of the second Sustainable Development Goal. This goal aims to end hunger and ensure access to safe and nutritious food for all.",
    "Sustainable Development Goal 3: Good Health and Well-being. Ensuring healthy lives and promoting well-being for all is the focus of the third goal. This includes efforts to reduce mortality from diseases, improve access to healthcare services, and promote mental health and well-being.",
    "Sustainable Development Goal 4: Quality Education. Goal four aims to ensure inclusive and equitable quality education and promote lifelong learning opportunities for all. This includes improving access to education, enhancing educational quality, and promoting gender equality.",
    "Sustainable Development Goal 5: Gender Equality. Achieving gender equality and empowering all women and girls is the fifth goal. This involves eliminating all forms of discrimination and violence against women, ensuring equal opportunities in leadership, and promoting women's full participation in decision-making processes.",
    "Sustainable Development Goal 6: Clean Water and Sanitation. Goal six focuses on ensuring availability and sustainable management of water and sanitation for all. This includes improving water quality, increasing water efficiency, and promoting hygiene practices.",
    "Sustainable Development Goal 7: Affordable and Clean Energy. Ensuring access to affordable, reliable, sustainable, and modern energy for all is the focus of the seventh goal. This includes promoting renewable energy sources and increasing energy efficiency.",
    "Sustainable Development Goal 8: Decent Work and Economic Growth. Promoting sustained, inclusive, and sustainable economic growth, full and productive employment, and decent work for all is the eighth goal. This involves creating more job opportunities, improving labor rights, and promoting entrepreneurship.",
    "Sustainable Development Goal 9: Industry, Innovation, and Infrastructure. Goal nine aims to build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation. This includes improving access to information and communication technology, enhancing research and development capacities, and upgrading infrastructure.",
    "Sustainable Development Goal 10: Reduced Inequalities. Reducing inequalities within and among countries is the focus of the tenth goal. This includes addressing income inequality, promoting the social, economic, and political inclusion of all, and ensuring equal opportunities for all.",
    "Sustainable Development Goal 11: Sustainable Cities and Communities. Goal eleven focuses on making cities and human settlements inclusive, safe, resilient, and sustainable. This involves improving access to affordable housing, promoting sustainable transport, and enhancing urban planning and management.",
    "Sustainable Development Goal 12: Responsible Consumption and Production. Ensuring sustainable consumption and production patterns is the twelfth goal. This includes promoting sustainable lifestyles, reducing waste generation, and promoting the use of environmentally friendly products and practices.",
    "Sustainable Development Goal 13: Climate Action. Taking urgent action to combat climate change and its impacts is the focus of the thirteenth goal. This involves strengthening resilience and adaptive capacity to climate-related disasters, integrating climate change measures into policies, and raising awareness on climate change.",
    "Sustainable Development Goal 14: Life Below Water. Conserving and sustainably using the oceans, seas, and marine resources for sustainable development is the focus of the fourteenth goal. This includes reducing marine pollution, protecting marine ecosystems, and sustainably managing fisheries.",
    "Sustainable Development Goal 15: Life on Land. Protecting, restoring, and promoting sustainable use of terrestrial ecosystems, sustainably managing forests, combating desertification, and halting biodiversity loss are the objectives of the fifteenth goal.",
    "Sustainable Development Goal 16: Peace, Justice, and Strong Institutions. Goal sixteen aims to promote peaceful and inclusive societies for sustainable development, provide access to justice for all, and build effective, accountable, and inclusive institutions at all levels.",
    "Sustainable Development Goal 17: Partnerships for the Goals. Strengthening the means of implementation and revitalizing the global partnership for sustainable development is the focus of the seventeenth goal. This includes enhancing international cooperation, promoting technology transfer, and increasing financial resources for developing countries.",
    "Achieving the Sustainable Development Goals requires a collaborative effort from governments, civil society, the private sector, and individuals. By working together, we can create a more sustainable and equitable world.",
    "The Sustainable Development Goals provide a comprehensive framework for addressing the world's most pressing challenges, from poverty and hunger to climate change and inequality.",
    "Achieving the Sustainable Development Goals is essential for ensuring a prosperous future for all. By investing in education, health, and sustainable infrastructure, we can create a world where no one is left behind.",
    "Goal four of the Sustainable Development Goals focuses on ensuring inclusive and quality education for all. By providing equal opportunities for education, we can empower individuals and communities to build a better future.",
    "Goal thirteen aims to take urgent action to combat climate change and its impacts. By reducing greenhouse gas emissions and promoting renewable energy, we can protect our planet for future generations.",
    "Sustainable Development Goal eight focuses on promoting sustained, inclusive, and sustainable economic growth. By creating job opportunities and supporting entrepreneurship, we can reduce poverty and inequality.",
    "Goal three of the Sustainable Development Goals aims to ensure healthy lives and promote well-being for all. By improving access to healthcare services and promoting mental health, we can enhance the overall quality of life.",
    "Sustainable Development Goal five focuses on achieving gender equality and empowering all women and girls. By eliminating gender-based discrimination and violence, we can create a more just and equal society.",
    "Goal seven of the Sustainable Development Goals aims to ensure access to affordable and clean energy for all. By investing in renewable energy sources and improving energy efficiency, we can combat climate change and reduce air pollution.",
    "Sustainable Development Goal twelve focuses on promoting responsible consumption and production. By reducing waste generation and adopting sustainable practices, we can protect the environment and conserve natural resources.",
    "Goal one of the Sustainable Development Goals aims to end poverty in all its forms. By ensuring access to basic services and economic opportunities, we can create a more equitable society.",
    "Sustainable Development Goal ten focuses on reducing inequalities within and among countries. By promoting social, economic, and political inclusion, we can create a more just and equal world.",
    "Achieving the Sustainable Development Goals requires the collective efforts of individuals, communities, and governments. By working together, we can create a better future for all.",
]

paragraphs = {"en": english_paragraphs, "fr": french_paragraphs}

corpus = ["ted", "conversation", "wikipedia", "hal"]


french_questions = [
    "Quel est le pourcentage de la population mondiale vivant en dessous du seuil de pauvreté ?",
    "Quelles sont les mesures mises en place pour réduire la pauvreté dans les pays en développement ?",
    "Comment les Objectifs de Développement Durable contribuent-ils à l'éradication de la pauvreté ?",
    "Combien de personnes dans le monde sont touchées par la faim ?",
    "Quels sont les principaux facteurs de l'insécurité alimentaire dans les pays en développement ?",
    "Comment l'agriculture durable peut-elle contribuer à la sécurité alimentaire et à l'éradication de la faim ?",
    "Quels sont les principaux défis en matière de santé dans les pays en développement ?",
    "Quels sont les facteurs qui contribuent aux maladies non transmissibles dans le monde ?",
    "Comment la promotion d'un mode de vie sain peut-elle contribuer au bien-être global de la population ?"
    "Comment l'Objectif de Développement Durable n°5 (Égalité des sexes) vise-t-il à éliminer les discriminations et les violences à l'égard des femmes et des filles ?",
    "Quels sont les moyens mis en œuvre pour assurer une éducation de qualité et inclusive ?",
    "Comment l'Objectif de Développement Durable n°6 (Eau propre et assainissement) vise-t-il à garantir l'accès à une eau potable et à des installations sanitaires pour tous ?",
    "Quels sont les objectifs principaux de l'Objectif de Développement Durable n°7 (Énergie propre et d'un coût abordable) ?",
    "Comment l'Objectif de Développement Durable n°9 (Industrie, innovation et infrastructure) favorise-t-il le développement durable et inclusif ?",
    "Quelles sont les mesures prises pour réduire les inégalités économiques et sociales au sein des pays (Objectif de Développement Durable n°10) ?",
    "Comment l'Objectif de Développement Durable n°11 (Villes et communautés durables) vise-t-il à rendre les villes plus inclusives, sûres et résilientes ?",
    "Quels sont les moyens déployés pour préserver et restaurer les écosystèmes terrestres (Objectif de Développement Durable n°15) ?",
    "Comment l'Objectif de Développement Durable n°16 (Paix, justice et institutions efficaces) contribue-t-il à la construction de sociétés pacifiques et inclusives ?",
    "Quelles sont les mesures prises pour promouvoir des modes de consommation et de production durables (Objectif de Développement Durable n°12) ?",
    "Comment l'Objectif de Développement Durable n°13 (Mesures relatives à la lutte contre les changements climatiques) vise-t-il à atténuer les effets du changement climatique et à favoriser la résilience face à ces changements ?",
    "Quels sont les moyens mis en œuvre pour promouvoir l'égalité des chances en matière d'emploi (Objectif de Développement Durable n°8)",
    "Comment l'Objectif de Développement Durable n°14 (Vie aquatique) vise-t-il à préserver et à utiliser de manière durable les océans, les mers et les ressources marines ?",
    "Quelles sont les mesures prises pour garantir l'accès à la justice pour tous (Objectif de Développement Durable n°16) ?",
    "Comment l'Objectif de Développement Durable n°17 (Partenariats pour la réalisation des objectifs) encourage-t-il la coopération internationale pour atteindre les Objectifs de Développement Durable ?",
]

english_questions = [
    "What are the key strategies being implemented to eradicate extreme poverty globally?",
    "How is the international community working towards reducing income inequalities within and among countries?",
    "What initiatives have been taken to ensure access to basic services and social protection for the poor and vulnerable populations?",
    "What are the main efforts being made to ensure food security and promote sustainable agriculture?",
    "How are governments addressing undernutrition and malnutrition among children and vulnerable populations?",
    "What are the key actions being taken to increase investment in rural infrastructure to enhance agricultural productivity and income?",
    "What measures are being taken to reduce maternal mortality and improve access to quality healthcare services for all?",
    "How is the international community addressing the growing prevalence of non-communicable diseases and mental health issues?",
    "What initiatives have been implemented to ensure universal access to sexual and reproductive health services and information?",
    "What steps are being taken to improve access to quality education for all, especially in marginalized communities?",
    "How are governments working to enhance the professional development and welfare of teachers worldwide?",
    "What initiatives have been implemented to promote inclusive and equitable education systems at all levels?",
    "What strategies are being implemented to eliminate gender-based discrimination and violence against women and girls?",
    "How are governments promoting women's economic empowerment and equal participation in decision-making processes?",
    "What initiatives have been taken to ensure universal access to sexual and reproductive health and rights?",
    "What steps are being taken to ensure access to safe and affordable drinking water for all?",
    "How is the international community working towards improving sanitation and hygiene practices globally?",
    "What initiatives have been implemented to protect and restore water-related ecosystems and promote sustainable water use?",
    "What measures are being taken to increase the share of renewable energy in the global energy mix?",
    "How are governments working to ensure universal access to modern, reliable, and sustainable energy services?",
    "What initiatives have been implemented to enhance energy efficiency and promote energy research and development?",
    "What steps are being taken to promote sustained, inclusive, and sustainable economic growth, full and productive employment, and decent work for all?",
    "How is the international community working towards eradicating forced labor, child labor, and modern slavery?",
    "What initiatives have been implemented to promote entrepreneurship, job creation, and sustainable tourism?",
    "What measures are being taken to promote inclusive and sustainable industrialization and foster innovation?",
    "How are governments working to enhance infrastructure development, particularly in less developed countries?",
    "What initiatives have been implemented to increase access to affordable, reliable, and sustainable energy for all?",
    "What strategies are being implemented to reduce inequalities within and among countries, particularly in terms of income, social protection, and opportunities?",
    "How is the international community working to empower and promote the social, economic, and political inclusion of all, irrespective of age, sex, disability, race, ethnicity, origin, religion, or economic or other status?",
    "What initiatives have been taken to ensure equal opportunity and reduce inequalities in access to education, healthcare, and other basic services?",
]

questions = {"en": english_questions, "fr": french_questions}


class QuickstartUser(HttpUser):
    # wait_time = between(10, 10)
    wait_time = constant_pacing(10)

    @task(1)
    def search(self):
        lang = random.choice(["en", "fr"])
        corpora = random.sample(corpus, random.randint(1, 4))
        self.client.post(
            "/api/v1/search/by_document",
            json={"corpora": corpora},
            params={
                "query": " ".join(random.sample(paragraphs[lang], 2)),
                "nb_results": 10,
            },
            name=f"/search/{lang}/{'&'.join(corpora)}",
        )

    @task(1)
    def search_slices(self):
        lang = random.choice(["en", "fr"])
        self.client.post(
            "/api/v1/search/by_slices",
            params={
                "query": " ".join(random.sample(paragraphs[lang], 2)),
                "nb_results": 10,
                "concatenate": True,
            },
            name=f"/search/by_slices/{lang}",
        )

    @task(2)
    def chat(self):
        lang = random.choice(["en", "fr"])
        with self.client.post(
            "/api/v1/qna/reformulate",
            params={"query": random.choice(questions[lang])},
            name=f"/qna/reformulate/{lang}",
        ) as resp1:
            try:
                reformulated_query = json.loads(resp1.text)["STANDALONE_QUESTION"]
            except json.decoder.JSONDecodeError:
                raise Exception("Question could not be reformulated")
        with self.client.post(
            "/api/v1/search/by_slices",
            params={"query": reformulated_query, "nb_results": 10, "concatenate": True},
            name=f"/search/by_slices/{lang}",
        ) as resp2:
            sources = json.loads(resp2.text)
        self.client.post(
            "/api/v1/qna/chat/answer",
            json={"sources": sources},
            params={"query": reformulated_query},
            name=f"/qna/chat/answer/{lang}",
        )
