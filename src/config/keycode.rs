use winit::{ElementState, MouseButton, VirtualKeyCode};

#[derive(Debug)]
pub struct Input {
    pub input: InputType,
    pub source_string: String,
    pub active: bool,
    pub status: InputStatus,
}

#[derive(Debug, PartialEq)]
pub enum InputType {
    Key(VirtualKeyCode),
    MouseKey(MouseButton),
    ScrollUp,
    ScrollDown,
    None,
}

#[derive(Debug, PartialEq)]
pub enum InputStatus {
    None,
    Down,
    Hold,
    Up,
}

impl Input {
    pub fn none(&self) -> bool {
        self.status == InputStatus::None
    }

    pub fn down(&self) -> bool {
        self.status == InputStatus::Down
    }

    pub fn hold(&self) -> bool {
        self.status == InputStatus::Hold
    }

    pub fn down_hold(&self) -> bool {
        self.status == InputStatus::Down || self.status == InputStatus::Hold
    }

    pub fn up(&self) -> bool {
        self.status == InputStatus::Up
    }

    pub fn update_key(&mut self, keycode: Option<VirtualKeyCode>, state: ElementState) {
        let matches = if let InputType::Key(key) = self.input {
            Some(key) == keycode
        } else {
            false
        };

        if matches {
            self.active = match state {
                ElementState::Pressed => true,
                ElementState::Released => false,
            };
        };
    }

    pub fn update_mouse(&mut self, button: MouseButton, state: ElementState) {
        let matches = InputType::MouseKey(button) == self.input;

        if matches {
            self.active = match state {
                ElementState::Pressed => true,
                ElementState::Released => false,
            };
        };
    }

    pub fn update_scroll(&mut self, delta: f32) {
        if (delta > 0.0 && self.input == InputType::ScrollUp)
            || (delta < 0.0 && self.input == InputType::ScrollDown)
        {
            self.active = true;
        } else if delta == 0.0
            && (self.input == InputType::ScrollUp || self.input == InputType::ScrollDown)
        {
            self.active = false;
        };
    }

    pub fn update_status(&mut self) {
        // none before but now pressed => down; else dont change
        // down and pressed => hold; else none
        // hold and not pressed => up; else dont change
        // up and not pressed => none; else down

        match self.status {
            InputStatus::None => if self.active {
                self.status = InputStatus::Down
            },
            InputStatus::Down => if self.active {
                self.status = InputStatus::Hold
            } else {
                self.status = InputStatus::None
            },
            InputStatus::Hold => if !self.active {
                self.status = InputStatus::Up
            },
            InputStatus::Up => if !self.active {
                self.status = InputStatus::None
            } else {
                self.status = InputStatus::Down
            },
        };
    }

    /*
    python 3

    file = open("keys.txt", "r")

    for line in file:
        line = line.strip()
        output = ""

        for index, char in enumerate(line):
            if char.istitle() & index != 0:
                output += "_"
            output += char

        print('"' + output.lower() + '"' + " => InputType::Key(VirtualKeyCode::" + line + "),")
    */
    pub fn from_str(field: Option<String>, location: &str, default: &str) -> Self {
        let input = field.unwrap_or_else(|| {
            ::warn!("{} is invalid, using default", location);
            default.to_owned()
        });

        let mut input_type = match &*input.to_lowercase() {
            // im sorry
            "mouse_left" => InputType::MouseKey(MouseButton::Left),
            "mouse_right" => InputType::MouseKey(MouseButton::Right),
            "mouse_middle" => InputType::MouseKey(MouseButton::Middle),
            "scroll_up" => InputType::ScrollUp,
            "scroll_down" => InputType::ScrollDown,
            "1" => InputType::Key(VirtualKeyCode::Key1), // removed "key"
            "2" => InputType::Key(VirtualKeyCode::Key2), // removed "key"
            "3" => InputType::Key(VirtualKeyCode::Key3), // removed "key"
            "4" => InputType::Key(VirtualKeyCode::Key4), // removed "key"
            "5" => InputType::Key(VirtualKeyCode::Key5), // removed "key"
            "6" => InputType::Key(VirtualKeyCode::Key6), // removed "key"
            "7" => InputType::Key(VirtualKeyCode::Key7), // removed "key"
            "8" => InputType::Key(VirtualKeyCode::Key8), // removed "key"
            "9" => InputType::Key(VirtualKeyCode::Key9), // removed "key"
            "0" => InputType::Key(VirtualKeyCode::Key0), // removed "key"
            "a" => InputType::Key(VirtualKeyCode::A),
            "b" => InputType::Key(VirtualKeyCode::B),
            "c" => InputType::Key(VirtualKeyCode::C),
            "d" => InputType::Key(VirtualKeyCode::D),
            "e" => InputType::Key(VirtualKeyCode::E),
            "f" => InputType::Key(VirtualKeyCode::F),
            "g" => InputType::Key(VirtualKeyCode::G),
            "h" => InputType::Key(VirtualKeyCode::H),
            "i" => InputType::Key(VirtualKeyCode::I),
            "j" => InputType::Key(VirtualKeyCode::J),
            "k" => InputType::Key(VirtualKeyCode::K),
            "l" => InputType::Key(VirtualKeyCode::L),
            "m" => InputType::Key(VirtualKeyCode::M),
            "n" => InputType::Key(VirtualKeyCode::N),
            "o" => InputType::Key(VirtualKeyCode::O),
            "p" => InputType::Key(VirtualKeyCode::P),
            "q" => InputType::Key(VirtualKeyCode::Q),
            "r" => InputType::Key(VirtualKeyCode::R),
            "s" => InputType::Key(VirtualKeyCode::S),
            "t" => InputType::Key(VirtualKeyCode::T),
            "u" => InputType::Key(VirtualKeyCode::U),
            "v" => InputType::Key(VirtualKeyCode::V),
            "w" => InputType::Key(VirtualKeyCode::W),
            "x" => InputType::Key(VirtualKeyCode::X),
            "y" => InputType::Key(VirtualKeyCode::Y),
            "z" => InputType::Key(VirtualKeyCode::Z),
            "escape" => InputType::Key(VirtualKeyCode::Escape),
            "f1" => InputType::Key(VirtualKeyCode::F1),
            "f2" => InputType::Key(VirtualKeyCode::F2),
            "f3" => InputType::Key(VirtualKeyCode::F3),
            "f4" => InputType::Key(VirtualKeyCode::F4),
            "f5" => InputType::Key(VirtualKeyCode::F5),
            "f6" => InputType::Key(VirtualKeyCode::F6),
            "f7" => InputType::Key(VirtualKeyCode::F7),
            "f8" => InputType::Key(VirtualKeyCode::F8),
            "f9" => InputType::Key(VirtualKeyCode::F9),
            "f10" => InputType::Key(VirtualKeyCode::F10),
            "f11" => InputType::Key(VirtualKeyCode::F11),
            "f12" => InputType::Key(VirtualKeyCode::F12),
            "f13" => InputType::Key(VirtualKeyCode::F13),
            "f14" => InputType::Key(VirtualKeyCode::F14),
            "f15" => InputType::Key(VirtualKeyCode::F15),
            "snapshot" => InputType::Key(VirtualKeyCode::Snapshot),
            "scroll" => InputType::Key(VirtualKeyCode::Scroll),
            "pause" => InputType::Key(VirtualKeyCode::Pause),
            "insert" => InputType::Key(VirtualKeyCode::Insert),
            "home" => InputType::Key(VirtualKeyCode::Home),
            "delete" => InputType::Key(VirtualKeyCode::Delete),
            "end" => InputType::Key(VirtualKeyCode::End),
            "pagedown" => InputType::Key(VirtualKeyCode::PageDown),
            "pageup" => InputType::Key(VirtualKeyCode::PageUp),
            "left" => InputType::Key(VirtualKeyCode::Left),
            "up" => InputType::Key(VirtualKeyCode::Up),
            "right" => InputType::Key(VirtualKeyCode::Right),
            "down" => InputType::Key(VirtualKeyCode::Down),
            "back" => InputType::Key(VirtualKeyCode::Back),
            "return" => InputType::Key(VirtualKeyCode::Return),
            "space" => InputType::Key(VirtualKeyCode::Space),
            "compose" => InputType::Key(VirtualKeyCode::Compose),
            "caret" => InputType::Key(VirtualKeyCode::Caret),
            "numlock" => InputType::Key(VirtualKeyCode::Numlock),
            "numpad0" => InputType::Key(VirtualKeyCode::Numpad0),
            "numpad1" => InputType::Key(VirtualKeyCode::Numpad1),
            "numpad2" => InputType::Key(VirtualKeyCode::Numpad2),
            "numpad3" => InputType::Key(VirtualKeyCode::Numpad3),
            "numpad4" => InputType::Key(VirtualKeyCode::Numpad4),
            "numpad5" => InputType::Key(VirtualKeyCode::Numpad5),
            "numpad6" => InputType::Key(VirtualKeyCode::Numpad6),
            "numpad7" => InputType::Key(VirtualKeyCode::Numpad7),
            "numpad8" => InputType::Key(VirtualKeyCode::Numpad8),
            "numpad9" => InputType::Key(VirtualKeyCode::Numpad9),
            "abntc1" => InputType::Key(VirtualKeyCode::AbntC1),
            "abntc2" => InputType::Key(VirtualKeyCode::AbntC2),
            "add" => InputType::Key(VirtualKeyCode::Add),
            "apostrophe" => InputType::Key(VirtualKeyCode::Apostrophe),
            "apps" => InputType::Key(VirtualKeyCode::Apps),
            "at" => InputType::Key(VirtualKeyCode::At),
            "ax" => InputType::Key(VirtualKeyCode::Ax),
            "backslash" => InputType::Key(VirtualKeyCode::Backslash),
            "calculator" => InputType::Key(VirtualKeyCode::Calculator),
            "capital" => InputType::Key(VirtualKeyCode::Capital),
            "colon" => InputType::Key(VirtualKeyCode::Colon),
            "comma" => InputType::Key(VirtualKeyCode::Comma),
            "convert" => InputType::Key(VirtualKeyCode::Convert),
            "decimal" => InputType::Key(VirtualKeyCode::Decimal),
            "divide" => InputType::Key(VirtualKeyCode::Divide),
            "equals" => InputType::Key(VirtualKeyCode::Equals),
            "grave" => InputType::Key(VirtualKeyCode::Grave),
            "kana" => InputType::Key(VirtualKeyCode::Kana),
            "kanji" => InputType::Key(VirtualKeyCode::Kanji),
            "l_alt" => InputType::Key(VirtualKeyCode::LAlt),
            "l_bracket" => InputType::Key(VirtualKeyCode::LBracket),
            "l_control" => InputType::Key(VirtualKeyCode::LControl),
            "l_shift" => InputType::Key(VirtualKeyCode::LShift),
            "l_win" => InputType::Key(VirtualKeyCode::LWin),
            "mail" => InputType::Key(VirtualKeyCode::Mail),
            "media_select" => InputType::Key(VirtualKeyCode::MediaSelect),
            "media_stop" => InputType::Key(VirtualKeyCode::MediaStop),
            "minus" => InputType::Key(VirtualKeyCode::Minus),
            "multiply" => InputType::Key(VirtualKeyCode::Multiply),
            "mute" => InputType::Key(VirtualKeyCode::Mute),
            "mycomputer" => InputType::Key(VirtualKeyCode::MyComputer),
            "navigateforward" => InputType::Key(VirtualKeyCode::NavigateForward),
            "navigatebackward" => InputType::Key(VirtualKeyCode::NavigateBackward),
            "nexttrack" => InputType::Key(VirtualKeyCode::NextTrack),
            "noconvert" => InputType::Key(VirtualKeyCode::NoConvert),
            "numpadcomma" => InputType::Key(VirtualKeyCode::NumpadComma),
            "numpadenter" => InputType::Key(VirtualKeyCode::NumpadEnter),
            "numpadequals" => InputType::Key(VirtualKeyCode::NumpadEquals),
            "o_em102" => InputType::Key(VirtualKeyCode::OEM102),
            "period" => InputType::Key(VirtualKeyCode::Period),
            "playpause" => InputType::Key(VirtualKeyCode::PlayPause),
            "power" => InputType::Key(VirtualKeyCode::Power),
            "prevtrack" => InputType::Key(VirtualKeyCode::PrevTrack),
            "r_alt" => InputType::Key(VirtualKeyCode::RAlt),
            "r_bracket" => InputType::Key(VirtualKeyCode::RBracket),
            "r_control" => InputType::Key(VirtualKeyCode::RControl),
            "r_shift" => InputType::Key(VirtualKeyCode::RShift),
            "r_win" => InputType::Key(VirtualKeyCode::RWin),
            "semicolon" => InputType::Key(VirtualKeyCode::Semicolon),
            "slash" => InputType::Key(VirtualKeyCode::Slash),
            "sleep" => InputType::Key(VirtualKeyCode::Sleep),
            "stop" => InputType::Key(VirtualKeyCode::Stop),
            "subtract" => InputType::Key(VirtualKeyCode::Subtract),
            "sysrq" => InputType::Key(VirtualKeyCode::Sysrq),
            "tab" => InputType::Key(VirtualKeyCode::Tab),
            "underline" => InputType::Key(VirtualKeyCode::Underline),
            "unlabeled" => InputType::Key(VirtualKeyCode::Unlabeled),
            "volumedown" => InputType::Key(VirtualKeyCode::VolumeDown),
            "volumeup" => InputType::Key(VirtualKeyCode::VolumeUp),
            "wake" => InputType::Key(VirtualKeyCode::Wake),
            "web_back" => InputType::Key(VirtualKeyCode::WebBack),
            "web_favorites" => InputType::Key(VirtualKeyCode::WebFavorites),
            "web_forward" => InputType::Key(VirtualKeyCode::WebForward),
            "web_home" => InputType::Key(VirtualKeyCode::WebHome),
            "web_refresh" => InputType::Key(VirtualKeyCode::WebRefresh),
            "web_search" => InputType::Key(VirtualKeyCode::WebSearch),
            "web_stop" => InputType::Key(VirtualKeyCode::WebStop),
            "yen" => InputType::Key(VirtualKeyCode::Yen),
            "copy" => InputType::Key(VirtualKeyCode::Copy),
            "paste" => InputType::Key(VirtualKeyCode::Paste),
            "cut" => InputType::Key(VirtualKeyCode::Cut),
            _ => InputType::None,
        };

        if input_type == InputType::None {
            ::warn!(
                "{} - \"{}\" isn't recognized, using default",
                location,
                input
            );

            input_type = Self::from_str(Some(default.to_owned()), location, default).input;
        }

        Input {
            input: input_type,
            source_string: input,
            active: false,
            status: InputStatus::None,
        }
    }
    /*
    python 3

    file = open("keys.txt", "r")

    for line in file:
        line = line.strip()

        string, defi = line.split(" => ")
        defi = defi[:-1]
        
        print(defi + " => " + string + ",")
    */
    pub fn to_string(&self) -> String {
        match self.input {
            InputType::MouseKey(MouseButton::Left) => "mouse_left",
            InputType::MouseKey(MouseButton::Right) => "mouse_right",
            InputType::MouseKey(MouseButton::Middle) => "mouse_middle",
            InputType::ScrollUp => "scroll_up",
            InputType::ScrollDown => "scroll_down",
            InputType::Key(VirtualKeyCode::Key1) => "1",
            InputType::Key(VirtualKeyCode::Key2) => "2",
            InputType::Key(VirtualKeyCode::Key3) => "3",
            InputType::Key(VirtualKeyCode::Key4) => "4",
            InputType::Key(VirtualKeyCode::Key5) => "5",
            InputType::Key(VirtualKeyCode::Key6) => "6",
            InputType::Key(VirtualKeyCode::Key7) => "7",
            InputType::Key(VirtualKeyCode::Key8) => "8",
            InputType::Key(VirtualKeyCode::Key9) => "9",
            InputType::Key(VirtualKeyCode::Key0) => "0",
            InputType::Key(VirtualKeyCode::A) => "a",
            InputType::Key(VirtualKeyCode::B) => "b",
            InputType::Key(VirtualKeyCode::C) => "c",
            InputType::Key(VirtualKeyCode::D) => "d",
            InputType::Key(VirtualKeyCode::E) => "e",
            InputType::Key(VirtualKeyCode::F) => "f",
            InputType::Key(VirtualKeyCode::G) => "g",
            InputType::Key(VirtualKeyCode::H) => "h",
            InputType::Key(VirtualKeyCode::I) => "i",
            InputType::Key(VirtualKeyCode::J) => "j",
            InputType::Key(VirtualKeyCode::K) => "k",
            InputType::Key(VirtualKeyCode::L) => "l",
            InputType::Key(VirtualKeyCode::M) => "m",
            InputType::Key(VirtualKeyCode::N) => "n",
            InputType::Key(VirtualKeyCode::O) => "o",
            InputType::Key(VirtualKeyCode::P) => "p",
            InputType::Key(VirtualKeyCode::Q) => "q",
            InputType::Key(VirtualKeyCode::R) => "r",
            InputType::Key(VirtualKeyCode::S) => "s",
            InputType::Key(VirtualKeyCode::T) => "t",
            InputType::Key(VirtualKeyCode::U) => "u",
            InputType::Key(VirtualKeyCode::V) => "v",
            InputType::Key(VirtualKeyCode::W) => "w",
            InputType::Key(VirtualKeyCode::X) => "x",
            InputType::Key(VirtualKeyCode::Y) => "y",
            InputType::Key(VirtualKeyCode::Z) => "z",
            InputType::Key(VirtualKeyCode::Escape) => "escape",
            InputType::Key(VirtualKeyCode::F1) => "f1",
            InputType::Key(VirtualKeyCode::F2) => "f2",
            InputType::Key(VirtualKeyCode::F3) => "f3",
            InputType::Key(VirtualKeyCode::F4) => "f4",
            InputType::Key(VirtualKeyCode::F5) => "f5",
            InputType::Key(VirtualKeyCode::F6) => "f6",
            InputType::Key(VirtualKeyCode::F7) => "f7",
            InputType::Key(VirtualKeyCode::F8) => "f8",
            InputType::Key(VirtualKeyCode::F9) => "f9",
            InputType::Key(VirtualKeyCode::F10) => "f10",
            InputType::Key(VirtualKeyCode::F11) => "f11",
            InputType::Key(VirtualKeyCode::F12) => "f12",
            InputType::Key(VirtualKeyCode::F13) => "f13",
            InputType::Key(VirtualKeyCode::F14) => "f14",
            InputType::Key(VirtualKeyCode::F15) => "f15",
            InputType::Key(VirtualKeyCode::Snapshot) => "snapshot",
            InputType::Key(VirtualKeyCode::Scroll) => "scroll",
            InputType::Key(VirtualKeyCode::Pause) => "pause",
            InputType::Key(VirtualKeyCode::Insert) => "insert",
            InputType::Key(VirtualKeyCode::Home) => "home",
            InputType::Key(VirtualKeyCode::Delete) => "delete",
            InputType::Key(VirtualKeyCode::End) => "end",
            InputType::Key(VirtualKeyCode::PageDown) => "pagedown",
            InputType::Key(VirtualKeyCode::PageUp) => "pageup",
            InputType::Key(VirtualKeyCode::Left) => "left",
            InputType::Key(VirtualKeyCode::Up) => "up",
            InputType::Key(VirtualKeyCode::Right) => "right",
            InputType::Key(VirtualKeyCode::Down) => "down",
            InputType::Key(VirtualKeyCode::Back) => "back",
            InputType::Key(VirtualKeyCode::Return) => "return",
            InputType::Key(VirtualKeyCode::Space) => "space",
            InputType::Key(VirtualKeyCode::Compose) => "compose",
            InputType::Key(VirtualKeyCode::Caret) => "caret",
            InputType::Key(VirtualKeyCode::Numlock) => "numlock",
            InputType::Key(VirtualKeyCode::Numpad0) => "numpad0",
            InputType::Key(VirtualKeyCode::Numpad1) => "numpad1",
            InputType::Key(VirtualKeyCode::Numpad2) => "numpad2",
            InputType::Key(VirtualKeyCode::Numpad3) => "numpad3",
            InputType::Key(VirtualKeyCode::Numpad4) => "numpad4",
            InputType::Key(VirtualKeyCode::Numpad5) => "numpad5",
            InputType::Key(VirtualKeyCode::Numpad6) => "numpad6",
            InputType::Key(VirtualKeyCode::Numpad7) => "numpad7",
            InputType::Key(VirtualKeyCode::Numpad8) => "numpad8",
            InputType::Key(VirtualKeyCode::Numpad9) => "numpad9",
            InputType::Key(VirtualKeyCode::AbntC1) => "abntc1",
            InputType::Key(VirtualKeyCode::AbntC2) => "abntc2",
            InputType::Key(VirtualKeyCode::Add) => "add",
            InputType::Key(VirtualKeyCode::Apostrophe) => "apostrophe",
            InputType::Key(VirtualKeyCode::Apps) => "apps",
            InputType::Key(VirtualKeyCode::At) => "at",
            InputType::Key(VirtualKeyCode::Ax) => "ax",
            InputType::Key(VirtualKeyCode::Backslash) => "backslash",
            InputType::Key(VirtualKeyCode::Calculator) => "calculator",
            InputType::Key(VirtualKeyCode::Capital) => "capital",
            InputType::Key(VirtualKeyCode::Colon) => "colon",
            InputType::Key(VirtualKeyCode::Comma) => "comma",
            InputType::Key(VirtualKeyCode::Convert) => "convert",
            InputType::Key(VirtualKeyCode::Decimal) => "decimal",
            InputType::Key(VirtualKeyCode::Divide) => "divide",
            InputType::Key(VirtualKeyCode::Equals) => "equals",
            InputType::Key(VirtualKeyCode::Grave) => "grave",
            InputType::Key(VirtualKeyCode::Kana) => "kana",
            InputType::Key(VirtualKeyCode::Kanji) => "kanji",
            InputType::Key(VirtualKeyCode::LAlt) => "l_alt",
            InputType::Key(VirtualKeyCode::LBracket) => "l_bracket",
            InputType::Key(VirtualKeyCode::LControl) => "l_control",
            InputType::Key(VirtualKeyCode::LShift) => "l_shift",
            InputType::Key(VirtualKeyCode::LWin) => "l_win",
            InputType::Key(VirtualKeyCode::Mail) => "mail",
            InputType::Key(VirtualKeyCode::MediaSelect) => "media_select",
            InputType::Key(VirtualKeyCode::MediaStop) => "media_stop",
            InputType::Key(VirtualKeyCode::Minus) => "minus",
            InputType::Key(VirtualKeyCode::Multiply) => "multiply",
            InputType::Key(VirtualKeyCode::Mute) => "mute",
            InputType::Key(VirtualKeyCode::MyComputer) => "mycomputer",
            InputType::Key(VirtualKeyCode::NavigateForward) => "navigateforward",
            InputType::Key(VirtualKeyCode::NavigateBackward) => "navigatebackward",
            InputType::Key(VirtualKeyCode::NextTrack) => "nexttrack",
            InputType::Key(VirtualKeyCode::NoConvert) => "noconvert",
            InputType::Key(VirtualKeyCode::NumpadComma) => "numpadcomma",
            InputType::Key(VirtualKeyCode::NumpadEnter) => "numpadenter",
            InputType::Key(VirtualKeyCode::NumpadEquals) => "numpadequals",
            InputType::Key(VirtualKeyCode::OEM102) => "o_em102",
            InputType::Key(VirtualKeyCode::Period) => "period",
            InputType::Key(VirtualKeyCode::PlayPause) => "playpause",
            InputType::Key(VirtualKeyCode::Power) => "power",
            InputType::Key(VirtualKeyCode::PrevTrack) => "prevtrack",
            InputType::Key(VirtualKeyCode::RAlt) => "r_alt",
            InputType::Key(VirtualKeyCode::RBracket) => "r_bracket",
            InputType::Key(VirtualKeyCode::RControl) => "r_control",
            InputType::Key(VirtualKeyCode::RShift) => "r_shift",
            InputType::Key(VirtualKeyCode::RWin) => "r_win",
            InputType::Key(VirtualKeyCode::Semicolon) => "semicolon",
            InputType::Key(VirtualKeyCode::Slash) => "slash",
            InputType::Key(VirtualKeyCode::Sleep) => "sleep",
            InputType::Key(VirtualKeyCode::Stop) => "stop",
            InputType::Key(VirtualKeyCode::Subtract) => "subtract",
            InputType::Key(VirtualKeyCode::Sysrq) => "sysrq",
            InputType::Key(VirtualKeyCode::Tab) => "tab",
            InputType::Key(VirtualKeyCode::Underline) => "underline",
            InputType::Key(VirtualKeyCode::Unlabeled) => "unlabeled",
            InputType::Key(VirtualKeyCode::VolumeDown) => "volumedown",
            InputType::Key(VirtualKeyCode::VolumeUp) => "volumeup",
            InputType::Key(VirtualKeyCode::Wake) => "wake",
            InputType::Key(VirtualKeyCode::WebBack) => "web_back",
            InputType::Key(VirtualKeyCode::WebFavorites) => "web_favorites",
            InputType::Key(VirtualKeyCode::WebForward) => "web_forward",
            InputType::Key(VirtualKeyCode::WebHome) => "web_home",
            InputType::Key(VirtualKeyCode::WebRefresh) => "web_refresh",
            InputType::Key(VirtualKeyCode::WebSearch) => "web_search",
            InputType::Key(VirtualKeyCode::WebStop) => "web_stop",
            InputType::Key(VirtualKeyCode::Yen) => "yen",
            InputType::Key(VirtualKeyCode::Copy) => "copy",
            InputType::Key(VirtualKeyCode::Paste) => "paste",
            InputType::Key(VirtualKeyCode::Cut) => "cut",
            _ => "",
        }.to_uppercase()
    }
}
