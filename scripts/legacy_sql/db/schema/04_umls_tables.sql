BEGIN;

CREATE TABLE IF NOT EXISTS umls.mrconso (
    aui TEXT PRIMARY KEY,
    cui TEXT NOT NULL,
    lat TEXT,
    ts TEXT,
    lui TEXT,
    stt TEXT,
    sui TEXT,
    ispref TEXT,
    saui TEXT,
    scui TEXT,
    sdui TEXT,
    sab TEXT,
    tty TEXT,
    code TEXT,
    str TEXT,
    srl TEXT,
    suppress TEXT,
    cvf TEXT
);

CREATE TABLE IF NOT EXISTS umls.mrrel (
    mrrel_id BIGSERIAL PRIMARY KEY,
    cui1 TEXT,
    aui1 TEXT,
    stype1 TEXT,
    rel TEXT,
    cui2 TEXT,
    aui2 TEXT,
    stype2 TEXT,
    rela TEXT,
    rui TEXT,
    srui TEXT,
    sab TEXT,
    sl TEXT,
    rg TEXT,
    dir TEXT,
    suppress TEXT,
    cvf TEXT
);

CREATE TABLE IF NOT EXISTS umls.mrsty (
    cui TEXT NOT NULL,
    tui TEXT NOT NULL,
    stn TEXT,
    sty TEXT,
    atui TEXT,
    cvf TEXT,
    PRIMARY KEY (cui, tui)
);

CREATE TABLE IF NOT EXISTS umls.mrdef (
    cui TEXT NOT NULL,
    aui TEXT NOT NULL,
    atui TEXT,
    satui TEXT,
    sab TEXT,
    definition TEXT,
    suppress TEXT,
    cvf TEXT
);

COMMIT;

-- MRMAP table for UMLS mapping
CREATE TABLE IF NOT EXISTS umls.mrmap (
    mapsetcui TEXT NOT NULL,
    mapsetsab TEXT NOT NULL,
    mapsubsetid TEXT,
    maprank INTEGER,
    mapid TEXT NOT NULL,
    mapsid TEXT,
    fromid TEXT NOT NULL,
    fromsid TEXT,
    fromexpr TEXT NOT NULL,
    fromtype TEXT NOT NULL,
    fromrule TEXT,
    fromres TEXT,
    rel TEXT NOT NULL,
    rela TEXT,
    toid TEXT,
    tosid TEXT,
    toexpr TEXT,
    totype TEXT,
    torule TEXT,
    tores TEXT,
    maprule TEXT,
    mapres TEXT,
    maptype TEXT,
    mapatn TEXT,
    mapatv TEXT,
    cvf INTEGER
    -- No primary key: MRMAP is a mapping table, may not have unique rows
);
