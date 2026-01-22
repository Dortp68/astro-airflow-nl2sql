CREATE TABLE IF NOT EXISTS "заказы" (
	"Дата операции" DATE, 
	"ИНН производителя" BIGINT, 
	"Наименование производителя" TEXT, 
	"ИНН продавца" BIGINT, 
	"Наименование продавца" TEXT, 
	"ИНН покупателя" TEXT, 
	"Наименование покупателя" TEXT, 
	"ТНВЭД" BIGINT, 
	"GTIN" BIGINT, 
	"Наименование товара" TEXT, 
	"Тип операции" TEXT, 
	"Объем товара, шт." INTEGER, 
	"Единицы измерения" TEXT, 
	"Объем товара, масса" numeric(12, 3)
);

CREATE TABLE IF NOT EXISTS "товары" (
	"ID товара" SERIAL NOT NULL, 
	"Наименование товара" TEXT NOT NULL, 
	"Категория" TEXT, 
	"Код ТНВЭД" BIGINT, 
	"GTIN" BIGINT, 
	"Единица измерения" TEXT, 
	"Описание" TEXT
);

CREATE TABLE IF NOT EXISTS "склады" (
	"ID склада" SERIAL NOT NULL, 
	"Расположение" TEXT, 
	"ID товара" INTEGER, 
	"Остаток (кол-во)" INTEGER, 
	"Остаток (масса), кг" numeric(12, 3), 
	"Последнее обновление" TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS "операции" (
	"ID операции" SERIAL NOT NULL, 
	"Дата операции" DATE NOT NULL, 
	"Тип операции" TEXT, 
	"ID товара" INTEGER, 
	"ID продавца" INTEGER, 
	"ID покупателя" INTEGER, 
	"Количество" INTEGER, 
	"Единица измерения" TEXT, 
	"Масса, кг" numeric(12, 3), 
	"Примечание" TEXT
);

CREATE TABLE IF NOT EXISTS "контрагенты" (
	"ID контрагента" SERIAL NOT NULL, 
	"ИНН" BIGINT NOT NULL, 
	"Наименование" TEXT NOT NULL, 
	"Тип контрагента" TEXT, 
	"Страна" TEXT
);

CREATE TABLE IF NOT EXISTS "поставщики" (
    "ID поставщика" SERIAL NOT NULL,
    "ИНН поставщика" BIGINT,
    "Наименование поставщика" TEXT,
    "Страна" TEXT,
    "Тип сырья" TEXT,
    "Контактное лицо" TEXT,
    "Email" TEXT
);

CREATE TABLE IF NOT EXISTS "производственные_партии" (
    "ID партии" SERIAL NOT NULL,
    "ID товара" INTEGER,
    "Дата производства" DATE,
    "Количество произведено" INTEGER,
    "Масса произведено, кг" numeric(12, 3),
    "Склад" TEXT
);

CREATE TABLE IF NOT EXISTS "отгрузки" (
    "ID отгрузки" SERIAL NOT NULL,
    "Дата отгрузки" DATE,
    "ID заказа" INTEGER,
    "Склад" TEXT,
    "Статус отгрузки" TEXT,
    "Транспортная компания" TEXT
);