//! Undo/redo via the command pattern. Every edit in the app goes through `History::execute`.

pub trait Command<S> {
    fn name(&self) -> &str;
    fn apply(&mut self, state: &mut S);
    fn revert(&mut self, state: &mut S);
}

pub struct History<S> {
    undo: Vec<Box<dyn Command<S>>>,
    redo: Vec<Box<dyn Command<S>>>,
}

impl<S> Default for History<S> {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
}

impl<S> History<S> {
    pub fn execute(&mut self, mut cmd: Box<dyn Command<S>>, state: &mut S) {
        cmd.apply(state);
        self.undo.push(cmd);
        self.redo.clear();
    }
    pub fn undo(&mut self, state: &mut S) -> bool {
        match self.undo.pop() {
            Some(mut c) => {
                c.revert(state);
                self.redo.push(c);
                true
            }
            None => false,
        }
    }
    pub fn redo(&mut self, state: &mut S) -> bool {
        match self.redo.pop() {
            Some(mut c) => {
                c.apply(state);
                self.undo.push(c);
                true
            }
            None => false,
        }
    }
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Add(i32);
    impl Command<i32> for Add {
        fn name(&self) -> &str {
            "add"
        }
        fn apply(&mut self, s: &mut i32) {
            *s += self.0
        }
        fn revert(&mut self, s: &mut i32) {
            *s -= self.0
        }
    }
    #[test]
    fn undo_redo_roundtrip() {
        let (mut s, mut h) = (0, History::default());
        h.execute(Box::new(Add(5)), &mut s);
        h.execute(Box::new(Add(2)), &mut s);
        assert_eq!(s, 7);
        h.undo(&mut s);
        assert_eq!(s, 5);
        h.redo(&mut s);
        assert_eq!(s, 7);
        h.undo(&mut s);
        h.execute(Box::new(Add(1)), &mut s);
        assert!(!h.can_redo());
    }
}
