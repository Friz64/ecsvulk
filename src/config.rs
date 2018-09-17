use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
};
use logger::Logger;
use keycode::Input;
use ::toml;
use ::winit::{
    VirtualKeyCode, ElementState, MouseButton,
};

#[derive(Deserialize, Serialize, Debug)]
struct OptionConfig {
    controls:       Option<OptionConfigControls>,
}

impl Default for OptionConfig {
    fn default() -> Self {
        OptionConfig {
            controls: Some(OptionConfigControls {
                movement: Some(OptionConfigControlsMovement {
                    forwards:       Some("w".to_uppercase()),
                    backwards:      Some("s".to_uppercase()),
                    left:           Some("a".to_uppercase()),
                    right:          Some("d".to_uppercase()),
                    speed_up:       Some("scroll_up".to_uppercase()),
                    speed_down:     Some("scroll_down".to_uppercase()),
                }),
                sensitivity: Some(OptionConfigControlsSensitivity {
                    mouse_speed:    Some(1.0),
                    movement_speed: Some(1.0),
                }),
            })
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct OptionConfigControls {
    movement:       Option<OptionConfigControlsMovement>,
    sensitivity:    Option<OptionConfigControlsSensitivity>,
}

impl Default for OptionConfigControls {
    fn default() -> Self {
        OptionConfigControls {
            movement: Some(OptionConfigControlsMovement {
                forwards:       Some("w".to_uppercase()),
                backwards:      Some("s".to_uppercase()),
                left:           Some("a".to_uppercase()),
                right:          Some("d".to_uppercase()),
                speed_up:       Some("scroll_up".to_uppercase()),
                speed_down:     Some("scroll_down".to_uppercase()),
            }),
            sensitivity: Some(OptionConfigControlsSensitivity {
                mouse_speed:    Some(1.0),
                movement_speed: Some(1.0),
            }),
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct OptionConfigControlsMovement {
    forwards:       Option<String>,
    backwards:      Option<String>,
    left:           Option<String>,
    right:          Option<String>,
    speed_up:       Option<String>,
    speed_down:     Option<String>,
}

impl Default for OptionConfigControlsMovement {
    fn default() -> Self {
        OptionConfigControlsMovement {
            forwards:       Some("w".to_uppercase()),
            backwards:      Some("s".to_uppercase()),
            left:           Some("a".to_uppercase()),
            right:          Some("d".to_uppercase()),
            speed_up:       Some("scroll_up".to_uppercase()),
            speed_down:     Some("scroll_down".to_uppercase()),
        }
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct OptionConfigControlsSensitivity {
    mouse_speed:    Option<f32>,
    movement_speed: Option<f32>,
}

impl Default for OptionConfigControlsSensitivity {
    fn default() -> Self {
        OptionConfigControlsSensitivity {
            mouse_speed:    Some(1.0),
            movement_speed: Some(1.0),
        }
    }
}

// Actual version thats used by the user.
#[derive(Debug)]
pub struct Config {
    pub controls:       ConfigControls,
}

#[derive(Debug)]
pub struct ConfigControls {
    pub movement:       ConfigControlsMovement,
    pub sensitivity:    ConfigControlsSensitivity,
}

#[derive(Debug)]
pub struct ConfigControlsMovement {
    pub forwards:       Input,
    pub backwards:      Input,
    pub left:           Input,
    pub right:          Input,
    pub speed_up:       Input,
    pub speed_down:     Input,
}

#[derive(Debug)]
pub struct ConfigControlsSensitivity {
    pub mouse_speed:    f32,
    pub movement_speed: f32,
}

impl Config {
    pub fn new(logger: &mut Logger, confpath: &str) -> Self {
        let config = OptionConfig::new(logger, confpath);

        let config_controls = config
            .controls.unwrap_or_else(|| {
            logger.warning("ConfigChecker", "config.controls is invalid, using default");
            Default::default()
        });
        let config_controls_movement = config_controls
            .movement.unwrap_or_else(|| {
            logger.warning("ConfigChecker", "config.controls.movement is invalid, using default");
            Default::default()
        });
        let config_controls_sensitivity = config_controls
            .sensitivity.unwrap_or_else(|| {
            logger.warning("ConfigChecker", "config.controls.sensitivity is invalid, using default");
            Default::default()
        });

        let result = Config {
            controls: ConfigControls {
                movement: ConfigControlsMovement {
                    forwards: Input::from_str(config_controls_movement.forwards.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.forwards is invalid, using default");
                        String::from("w")
                    })).warn_none(logger, "config.controls.movement.forwards"),

                    backwards: Input::from_str(config_controls_movement.backwards.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.backwards is invalid, using default");
                        String::from("s")
                    })).warn_none(logger, "config.controls.movement.backwards"),

                    left: Input::from_str(config_controls_movement.left.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.left is invalid, using default");
                        String::from("a")
                    })).warn_none(logger, "config.controls.movement.left"),

                    right: Input::from_str(config_controls_movement.right.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.right is invalid, using default");
                        String::from("d")
                    })).warn_none(logger, "config.controls.movement.right"),

                    speed_up: Input::from_str(config_controls_movement.speed_up.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.speed_up is invalid, using default");
                        String::from("scroll_up")
                    })).warn_none(logger, "config.controls.movement.speed_up"),

                    speed_down: Input::from_str(config_controls_movement.speed_down.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.movement.speed_down is invalid, using default");
                        String::from("scroll_down")
                    })).warn_none(logger, "config.controls.movement.speed_down"),
                },
                sensitivity: ConfigControlsSensitivity {
                    mouse_speed: config_controls_sensitivity.mouse_speed.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.sensitivity.mouse_speed is invalid, using default");
                        1.0
                    }),

                    movement_speed: config_controls_sensitivity.movement_speed.unwrap_or_else(|| {
                        logger.warning("ConfigChecker", "config.controls.sensitivity.movement_speed is invalid, using default");
                        1.0
                    }),
                },
            },
        };

        logger.info("Config", "Config initialized");

        result
    }

    pub fn save(&self, logger: &mut Logger, config_path: &str) {
        let path = format!("./{}/{}", ::NAME, config_path);

        let reconstructed = OptionConfig {
            controls: Some(OptionConfigControls {
                movement: Some(OptionConfigControlsMovement {
                    forwards: Some(self.controls.movement.forwards.to_string()),
                    backwards: Some(self.controls.movement.backwards.to_string()),
                    left: Some(self.controls.movement.left.to_string()),
                    right: Some(self.controls.movement.right.to_string()),
                    speed_up: Some(self.controls.movement.speed_up.to_string()),
                    speed_down: Some(self.controls.movement.speed_down.to_string()),
                }),
                sensitivity: Some(OptionConfigControlsSensitivity {
                    mouse_speed: Some(self.controls.sensitivity.mouse_speed),
                    movement_speed: Some(self.controls.sensitivity.movement_speed),
                })
            })
        };
        let serialized = toml::to_string(&reconstructed)
            .unwrap_or_else(|err| {
                logger.warning("ConfigGen", err);
                String::new()
            });
        
        let mut file = OpenOptions::new()
            .write(true).open(&path)
            .unwrap_or_else(|err| logger.error("ConfigSave", err));

        // delete everything in file
        file.set_len(0)
            .unwrap_or_else(|err| logger.error("ConfigSave", err));

        file.write(serialized.as_bytes())
            .unwrap_or_else(|err| logger.error("ConfigSave", err));
    }

    pub fn update_keys(&mut self, keycode: Option<VirtualKeyCode>, state: ElementState) {
        self.controls.movement.forwards  .update_key(keycode, state);
        self.controls.movement.backwards .update_key(keycode, state);
        self.controls.movement.left      .update_key(keycode, state);
        self.controls.movement.right     .update_key(keycode, state);
        self.controls.movement.speed_up  .update_key(keycode, state);
        self.controls.movement.speed_down.update_key(keycode, state);
    }

    pub fn update_mouse(&mut self, button: MouseButton, state: ElementState) {
        self.controls.movement.forwards  .update_mouse(button, state);
        self.controls.movement.backwards .update_mouse(button, state);
        self.controls.movement.left      .update_mouse(button, state);
        self.controls.movement.right     .update_mouse(button, state);
        self.controls.movement.speed_up  .update_mouse(button, state);
        self.controls.movement.speed_down.update_mouse(button, state);
    }
    
    pub fn update_scroll(&mut self, delta: f32) {
        self.controls.movement.forwards  .update_scroll(delta);
        self.controls.movement.backwards .update_scroll(delta);
        self.controls.movement.left      .update_scroll(delta);
        self.controls.movement.right     .update_scroll(delta);
        self.controls.movement.speed_up  .update_scroll(delta);
        self.controls.movement.speed_down.update_scroll(delta);
    }

    pub fn update_status(&mut self) {
        self.controls.movement.forwards  .update_status();
        self.controls.movement.backwards .update_status();
        self.controls.movement.left      .update_status();
        self.controls.movement.right     .update_status();
        self.controls.movement.speed_up  .update_status();
        self.controls.movement.speed_down.update_status();
    }
}

impl OptionConfig {
    pub fn new(logger: &mut Logger, config_path: &str) -> Self {
        let path = format!("./{}/{}", ::NAME, config_path);
        let default = OptionConfig::default();

        let mut file = File::open(&path)
            .unwrap_or_else(|err| match err.kind() {
                io::ErrorKind::NotFound => {
                    logger.warning("ConfigMissing", "Config file missing, creating new one");

                    // file::create doesn't allow reading
                    // https://doc.rust-lang.org/std/fs/struct.OpenOptions.html
                    let mut file = OpenOptions::new()
                        .read(true).write(true).create(true).open(&path)
                        .unwrap_or_else(|err|
                            logger.error("ConfigCreate", err)
                        );

                    let serialized = toml::to_string(&default)
                        .unwrap_or_else(|err| {
                            logger.warning("ConfigGen", err);
                            String::new()
                        });

                    file.write(serialized.as_bytes())
                        .map_err(|err| {
                            logger.warning("ConfigWrite", err);
                        }).ok();

                    file
                },
                _ => logger.error("ConfigOpen", err),
            });

        // read the file from start because we just wrote to it
        // https://doc.rust-lang.org/std/io/trait.Seek.html#tymethod.seek
        file.seek(SeekFrom::Start(0))
            .unwrap_or_else(|err| {
                logger.warning("ConfigRead", err);
                Default::default()
            });

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .unwrap_or_else(|err| {
                logger.warning("ConfigRead", err);
                Default::default()
            });

        toml::from_str(&contents)
            .unwrap_or_else(|err| {
                logger.warning("ConfigDecode", err.to_string() + " - USING DEFAULT CONFIG INSTEAD");
                OptionConfig::default()
            })
    }
}