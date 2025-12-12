CREATE TABLE users
(
  user_id INT NOT NULL,
  username VARCHAR(64) NOT NULL,
  email VARCHAR(254) NOT NULL,
  preferred_theme VARCHAR(15) NOT NULL,
  created_at DATE NOT NULL,
  phone_number VARCHAR(20),
  saved_vis_num INT NOT NULL DEFAULT (0),
  PRIMARY KEY (user_id)
);

CREATE TABLE user_auth
(
  password VARCHAR(MAX) NOT NULL,
  token VARCHAR(MAX) NOT NULL,
  token_created_at DATE NOT NULL,
  user_id INT NOT NULL,
  PRIMARY KEY (password, user_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE user_post
(
  id INT NOT NULL,
  operation_type VARCHAR(30) NOT NULL,
  created_at DATE NOT NULL,
  status VARCHAR(15) NOT NULL,
  user_id INT NOT NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE users_saved_visuals
(
  saved_visual JSON NOT NULL,
  type VARCHAR(30) NOT NULL,
  updated_at DATE NOT NULL,
  user_id INT NOT NULL,
  PRIMARY KEY (saved_visual, user_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE label_correction
(
  image_path VARCHAR(256) NOT NULL,
  data_structure_type VARCHAR(30) NOT NULL,
  wrong_label JSON,
  correct_label JSON NOT NULL,
  created_at DATE NOT NULL,
  user_id INT NOT NULL,
  PRIMARY KEY (image_path),
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);