import os
import time
import random
from icrawler.builtin import BingImageCrawler

# Türkiye'nin En Popüler 100 Araç Kasası (Paspas Savar Filtreli Arama Terimleri İçin Hazır)
TOP_CARS = [
    # Fiat & Renault
    "Fiat Egea Sedan 2021", "Fiat Egea Cross 2022", "Fiat Egea Hatchback 2020", "Fiat Fiorino 2020", "Fiat Doblo 2018", "Fiat Linea 2015",
    "Renault Clio 5 2021", "Renault Clio 4 2016", "Renault Megane 4 Sedan 2020", "Renault Megane 3 2014", "Renault Symbol 2016", "Renault Kadjar 2018", "Renault Captur 2020", "Renault Fluence 2015",
    
    # Volkswagen Grubu
    "Volkswagen Passat B8 2018", "Volkswagen Passat B7 2013", "Volkswagen Golf 8 2021", "Volkswagen Golf 7 2016", "Volkswagen Polo 2020", "Volkswagen Jetta 2016", "Volkswagen Tiguan 2019", "Volkswagen T-Roc 2021", "Volkswagen Caddy 2021", "Volkswagen Transporter T6 2018",
    
    # Toyota & Honda
    "Toyota Corolla 2021", "Toyota Corolla 2016", "Toyota C-HR 2020", "Toyota Yaris 2021", "Toyota Hilux 2020", "Toyota Auris 2015",
    "Honda Civic FC5 2019", "Honda Civic FD6 2010", "Honda Civic FB7 2014", "Honda CR-V 2018",
    
    # Ford & Hyundai
    "Ford Focus 2016", "Ford Focus 2020", "Ford Fiesta 2015", "Ford Courier 2020", "Ford Transit Custom 2019", "Ford Puma 2021", "Ford Kuga 2020",
    "Hyundai i20 2021", "Hyundai i20 2016", "Hyundai i10 2020", "Hyundai Tucson 2022", "Hyundai Tucson 2017", "Hyundai Accent Blue 2016", "Hyundai Elantra 2018", "Hyundai Bayon 2022",
    
    # Peugeot & Dacia
    "Peugeot 3008 2020", "Peugeot 2008 2021", "Peugeot 208 2021", "Peugeot 308 2017", "Peugeot Rifter 2020",
    "Dacia Duster 2021", "Dacia Duster 2016", "Dacia Sandero Stepway 2021", "Dacia Logan 2015",
    
    # Opel & Skoda & Seat
    "Opel Astra J 2015", "Opel Astra K 2018", "Opel Corsa 2021", "Opel Mokka 2022", "Opel Insignia 2017",
    "Skoda Octavia 2021", "Skoda Octavia 2016", "Skoda Superb 2020", "Skoda Scala 2021", "Skoda Kamiq 2021",
    "Seat Leon 2016", "Seat Leon 2021", "Seat Ibiza 2018", "Seat Ateca 2020",
    
    # Nissan & Kia
    "Nissan Qashqai 2018", "Nissan Qashqai 2014", "Nissan Micra 2017", "Nissan Juke 2020",
    "Kia Sportage 2022", "Kia Sportage 2016", "Kia Ceed 2017", "Kia Rio 2018", "Kia Stonic 2021",
    
    # Premium: Audi, BMW, Mercedes
    "Audi A3 Sedan 2018", "Audi A4 B9 2017", "Audi A6 2019", "Audi Q3 2020",
    "BMW 3 Series F30 2016", "BMW 3 Series G20 2021", "BMW 5 Series F10 2015", "BMW 5 Series G30 2018", "BMW 1 Series 2017",
    "Mercedes Benz C-Class W205 2017", "Mercedes Benz E-Class W213 2018", "Mercedes Benz A-Class 2019", "Mercedes Benz CLA 2017",
    
    # Volvo, Chery, Togg, MG, Cupra, Mitsubishi, Mazda
    "Volvo XC40 2021", "Volvo XC90 2019", "Chery Tiggo 8 Pro 2023", "Chery Tiggo 7 Pro 2023", "Chery Omoda 5 2023",
    "Togg T10X 2023", "MG ZS 2022", "Cupra Formentor 2022", "Mitsubishi L200 2019", "Mazda 3 2017", "Suzuki Vitara 2020"
]

def sanitize_folder_name(car_name: str) -> str:
    return car_name.lower().replace(" ", "_").replace("-", "_")

def clean_small_files(directory, min_size_kb=20):
    """Paspas, ikon veya bozuk dosyaları (genelde küçük olurlar) temizler."""
    if not os.path.exists(directory): return
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if os.path.getsize(filepath) / 1024 < min_size_kb:
                os.remove(filepath)

def collect_data(images_per_car=100, base_data_dir="car_vision_project/data"):
    print(f"🚀 {len(TOP_CARS)} araç için OPTİMİZE hasat başladı. Hedef: {len(TOP_CARS) * images_per_car} görsel.")
    
    for car in TOP_CARS:
        folder_name = sanitize_folder_name(car)
        target_dir = os.path.join(base_data_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Temizlik yap ve mevcut dosya sayısını kontrol et
        clean_small_files(target_dir)
        current_count = len([n for n in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, n))])
        
        if current_count >= images_per_car:
            print(f"✅ [{car}] zaten tamam. Atlanıyor...")
            continue
            
        # Paspas ve iç mekan gelmesini engelleyen 'Exterior-Only' stratejileri
        search_strategies = [
            f"{car} car full body exterior photo",
            f"{car} car street view real photo",
            f"{car} car outside driving view",
            f"{car} car professional automotive photography"
        ]
        
        for keyword in search_strategies:
            existing = len([n for n in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, n))])
            if existing >= images_per_car: break
            
            remaining = images_per_car - existing
            print(f"⏳ [{car}] | Mevcut: {existing}/{images_per_car} | Arama: {keyword}")
            
            # Botun ban yemesini engellemek için downloader_threads'i 2 yaptık
            crawler = BingImageCrawler(storage={'root_dir': target_dir}, downloader_threads=2)
            
            try:
                # 'large' filtresini kaldırdık çünkü bazen kaliteli ama orta boy fotoları kaçırıyor
                crawler.crawl(keyword=keyword, filters={'type': 'photo'}, max_num=remaining, file_idx_offset='auto')
            except Exception as e:
                print(f"⚠️ Hata: {e}")
            
            clean_small_files(target_dir)
            # Anti-Ban bekleme süresini artırdık
            time.sleep(random.uniform(3.0, 6.0))
            
    print("\n🏁 Master Hasat Bitti! Şimdi 'data' klasörüne girip hızlıca bir göz atma vakti.")

if __name__ == "__main__":
    collect_data(images_per_car=100)